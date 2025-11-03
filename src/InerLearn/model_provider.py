# ==========================================
# model_provider.py — Modular Estimator Suite
# ==========================================

import os, pickle, hashlib, logging
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ==============================================================
# Model Persistence Utilities
# ==============================================================
def _hash_dataset(X, y):
    """Create a short hash fingerprint of dataset content for versioning."""
    m = hashlib.sha1()
    m.update(np.asarray(X).tobytes())
    m.update(np.asarray(y).tobytes())
    return m.hexdigest()[:10]


def save_model_cache(provider, cache_dir="cache/models", tag=None):
    """
    Save trained models to disk (both classical and deep) with dataset fingerprint.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset_hash = tag or _hash_dataset(provider.Xtr[:100], provider.ytr[:100])  # small subset to hash
    cache_path = os.path.join(cache_dir, f"model_provider_{dataset_hash}.pkl")

    # Separate PyTorch models (save with torch.save)
    deep_dir = os.path.join(cache_dir, f"deep_{dataset_hash}")
    os.makedirs(deep_dir, exist_ok=True)

    serializable_models = {}
    for name, model in provider.models.items():
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), os.path.join(deep_dir, f"{name}.pt"))
            serializable_models[name] = {"_type": "torch", "path": f"{deep_dir}/{name}.pt"}
        else:
            serializable_models[name] = model

    with open(cache_path, "wb") as f:
        pickle.dump({
            "models": serializable_models,
            "window": provider.window,
            "dataset_hash": dataset_hash
        }, f)
    logger.info(f"Model cache saved at {cache_path}")


def load_model_cache(cache_dir="cache/models", tag=None, device="cpu"):
    """
    Load a cached ModelProvider (classical + deep models).
    Returns None if cache not found.
    """
    if not os.path.exists(cache_dir):
        logger.warning("Model cache directory not found.")
        return None

    candidates = [f for f in os.listdir(cache_dir) if f.startswith("model_provider_")]
    if not candidates:
        logger.warning("No cached model providers found.")
        return None

    if tag:
        filename = f"model_provider_{tag}.pkl"
        cache_path = os.path.join(cache_dir, filename)
        if not os.path.exists(cache_path):
            logger.warning(f"Requested model cache '{filename}' not found.")
            return None
    else:
        cache_path = os.path.join(cache_dir, sorted(candidates)[-1])  # latest

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    models = {}
    for name, m in data["models"].items():
        if isinstance(m, dict) and m.get("_type") == "torch":
            path = m["path"]
            if "GRU" in name: model = GRUModel()
            elif "LSTM" in name and not "Bi" in name: model = LSTMModel()
            elif "BiLSTM" in name: model = BiLSTMModel()
            elif "CNN1D" in name: model = CNN1DModel()
            elif "CNNLSTM" in name: model = CNNLSTMModel()
            elif "Transformer" in name: model = TransformerModel()
            else:
                logger.warning(f"Unknown model type '{name}', skipping.")
                continue
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models[name] = model
        else:
            models[name] = m

    logger.info(f"Loaded model cache from {cache_path}")
    provider = ModelProvider(np.zeros((1, 6)), np.zeros((1, 3)), np.zeros((1, 6)), np.zeros((1, 3)))
    provider.models = models
    return provider


# ==============================================================
# Deep Architectures
# ==============================================================
class GRUModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x):
        y, _ = self.gru(x)
        return self.fc(y[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])


class CNN1DModel(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU()
        )
        self.fc = nn.Linear(256, 3)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).mean(dim=2)
        return self.fc(x)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_dim=6, embed_dim=16, hidden_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                               dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Linear(embed_dim, 3)
    def forward(self, x):
        x = self.input_proj(x)
        y = self.encoder(x)
        return self.fc(y[:, -1, :])


# ==============================================================
# Model Provider
# ==============================================================
class ModelProvider:
    def __init__(self, Xtr, ytr, Xte, yte, window=40):
        self.Xtr, self.ytr, self.Xte, self.yte, self.window = Xtr, ytr, Xte, yte, window
        self.models = {}
        self._init_classical_models()
        self._init_deep_models()

    def _init_classical_models(self):
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
        try:
            tabpfn_model = TabPFNRegressor(N_ensemble_configurations=4, ignore_pretraining_limits=True)
        except TypeError:
            tabpfn_model = TabPFNRegressor(ignore_pretraining_limits=True)

        self.models_classical = {
            "Ridge": Ridge(alpha=0.25),
            "SVR": MultiOutputRegressor(SVR(kernel="rbf", C=10, gamma="scale")),
            "KNN": MultiOutputRegressor(KNeighborsRegressor(n_neighbors=8, weights="distance")),
            "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=14, n_jobs=-1)),
            "ExtraTrees": MultiOutputRegressor(ExtraTreesRegressor(n_estimators=200, max_depth=None, n_jobs=-1)),
            "GradBoost": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5)),
            "AdaBoost": MultiOutputRegressor(AdaBoostRegressor(n_estimators=200, learning_rate=0.5)),
            "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=250, max_depth=8, learning_rate=0.05, subsample=0.9, n_jobs=-1, tree_method='hist')),
            "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=250, max_depth=-1, learning_rate=0.05, num_leaves=64, verbose=-1)),
            "CatBoost": MultiOutputRegressor(CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8, verbose=False)),
            "TabPFN": MultiOutputRegressor(tabpfn_model)
        }

    def _init_deep_models(self):
        self.models_deep = {
            "GRU": GRUModel(),
            "LSTM": LSTMModel(),
            "BiLSTM": BiLSTMModel(),
            "CNN1D": CNN1DModel(),
            "CNNLSTM": CNNLSTMModel(),
            "Transformer": TransformerModel()
        }

    def train_classical(self):
        for name, model in self.models_classical.items():
            logger.info(f"Training classical model: {name}")
            model.fit(self.Xtr, self.ytr)
        self.models.update(self.models_classical)
        logger.info("All classical models trained successfully.")

    def train_deep(self, epochs=30, batch_size=256):
        for name, model in self.models_deep.items():
            logger.info(f"Training deep model: {name}")
            self._train_seq_model(model, epochs, batch_size)
        self.models.update(self.models_deep)
        logger.info("All deep models trained successfully.")

    def _train_seq_model(self, model, epochs, batch_size):
        X_seq = self.Xtr.reshape(-1, self.window, 6)
        Ds = tud.TensorDataset(torch.tensor(X_seq, dtype=torch.float32),
                               torch.tensor(self.ytr, dtype=torch.float32))
        Dl = tud.DataLoader(Ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.MSELoss()

        for e in range(epochs):
            total = 0
            for xb, yb in Dl:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item()
            sched.step()
            logger.debug(f"{model.__class__.__name__}: epoch {e+1}/{epochs}, loss={total/len(Dl):.6f}")
        return model

    def evaluate_all(self):
        logger.info("Evaluating all models (validation MSE)...")
        val_table = []
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                yp = model.predict(self.Xte)
            else:
                X_seq = torch.tensor(self.Xte.reshape(-1, self.window, 6), dtype=torch.float32)
                with torch.no_grad():
                    yp = model(X_seq).numpy()
            mse = mean_squared_error(self.yte, yp)
            val_table.append((name, mse))
        val_table = sorted(val_table, key=lambda x: x[1])

        logger.info("===== VALIDATION MSE RANKING =====")
        for rank, (name, mse) in enumerate(val_table, 1):
            logger.info(f"{rank:2d}. {name:15s} → MSE {mse:.6f}")
        logger.info(f"Best model: {val_table[0][0]}")
        return val_table

    def get_model(self, name):
        return self.models.get(name, None)
