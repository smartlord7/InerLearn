"""
model_provider.py
=================
Unified model management module for IMU bias learning.
Handles classical + deep models, caching, training, and evaluation.
"""

import os, pickle, hashlib, torch, torch.nn as nn, torch.utils.data as tud
import numpy as np
import logging
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
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger("imu_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================================================
# Utility
# ============================================================
def _ensure_dir(path): os.makedirs(path, exist_ok=True)

def _hash_dataset(X, y):
    h = hashlib.sha1()
    h.update(np.asarray(X).tobytes())
    h.update(np.asarray(y).tobytes())
    return h.hexdigest()[:10]

def _make_windows(acc, gyro, window):
    X = np.hstack([acc, gyro])
    Xw = np.lib.stride_tricks.sliding_window_view(X, (window, X.shape[1]))
    return Xw.reshape(-1, window, X.shape[1])

# ============================================================
# Cache
# ============================================================
def save_model(model, name, dataset_hash, cache_dir="cache/models", window=None):
    _ensure_dir(cache_dir)
    meta_path = os.path.join(cache_dir, f"meta_{dataset_hash}.pkl")
    entry = {"name": name, "window": window, "dataset_hash": dataset_hash}

    if isinstance(model, nn.Module):
        deep_dir = os.path.join(cache_dir, f"deep_{dataset_hash}")
        _ensure_dir(deep_dir)
        path = os.path.join(deep_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)
        entry.update({"_type": "torch", "path": path})
    else:
        path = os.path.join(cache_dir, f"{name}_{dataset_hash}.pkl")
        with open(path, "wb") as f: pickle.dump(model, f)
        entry.update({"_type": "pickle", "path": path})

    meta = []
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            try: meta = pickle.load(f)
            except Exception: meta = []
    meta = [m for m in meta if m["name"] != name]
    meta.append(entry)
    with open(meta_path, "wb") as f: pickle.dump(meta, f)
    logger.info(f"Cached model '{name}' → {path}")

def load_cached_models(cache_dir="cache/models", tag=None, device="cpu"):
    if not os.path.exists(cache_dir): return {}
    metas = [f for f in os.listdir(cache_dir) if f.startswith("meta_")]
    if not metas: return {}
    fname = f"meta_{tag}.pkl" if tag else sorted(metas)[-1]
    meta_path = os.path.join(cache_dir, fname)
    with open(meta_path, "rb") as f: entries = pickle.load(f)
    models = {}
    for e in entries:
        try:
            if e["_type"] == "torch":
                clsmap = {
                    "GRU": GRUModel, "LSTM": LSTMModel, "BiLSTM": BiLSTMModel,
                    "CNN1D": CNN1DModel, "CNNLSTM": CNNLSTMModel, "Transformer": TransformerModel
                }
                m = clsmap[e["name"]]()
                m.load_state_dict(torch.load(e["path"], map_location=device))
                m.eval()
                models[e["name"]] = m
            else:
                with open(e["path"], "rb") as f: models[e["name"]] = pickle.load(f)
        except Exception as ex:
            logger.warning(f"Failed to load {e['name']}: {ex}")
    logger.info(f"Loaded {len(models)} cached models.")
    return models

# ============================================================
# Deep Models
# ============================================================
class GRUModel(nn.Module):
    def __init__(self, input_dim=6, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x): return self.fc(self.gru(x)[0][:, -1, :])

class LSTMModel(GRUModel):
    def __init__(self, input_dim=6, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x): return self.fc(self.lstm(x)[0][:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden*2, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, x): return self.fc(self.lstm(x)[0][:, -1, :])

class CNN1DModel(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim,64,3,padding=1), nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1), nn.ReLU(),
            nn.Conv1d(128,256,3,padding=1), nn.ReLU()
        )
        self.fc = nn.Linear(256,3)
    def forward(self,x): return self.fc(self.conv(x.transpose(1,2)).mean(dim=2))

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden=128):
        super().__init__()
        self.conv = nn.Conv1d(input_dim,64,3,padding=1)
        self.lstm = nn.LSTM(64,hidden,batch_first=True)
        self.fc = nn.Linear(hidden,3)
    def forward(self,x):
        x=torch.relu(self.conv(x.transpose(1,2))).transpose(1,2)
        return self.fc(self.lstm(x)[0][:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self,input_dim=6,embed_dim=16,hidden=128,nhead=4,nlayers=2):
        super().__init__()
        self.proj=nn.Linear(input_dim,embed_dim)
        enc_layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=nhead,
                                             dim_feedforward=hidden,batch_first=True)
        self.encoder=nn.TransformerEncoder(enc_layer,nlayers)
        self.fc=nn.Linear(embed_dim,3)
    def forward(self,x): return self.fc(self.encoder(self.proj(x))[:, -1, :])

# ============================================================
# Model Provider
# ============================================================
class ModelProvider:
    def __init__(self, Xtr, ytr, Xte, yte, window=40):
        self.Xtr, self.ytr, self.Xte, self.yte = Xtr, ytr, Xte, yte
        self.window = window
        self.dataset_hash = _hash_dataset(Xtr[:100], ytr[:100])
        self.models = {}
        self._init_models()

    def _init_models(self):
        self.models_classical = {
            "Ridge": Ridge(alpha=0.25),
            "SVR": MultiOutputRegressor(SVR(C=10, kernel="rbf", gamma="scale")),
            "KNN": MultiOutputRegressor(KNeighborsRegressor(n_neighbors=8, weights="distance")),
            "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=14, n_jobs=-1)),
            "ExtraTrees": MultiOutputRegressor(ExtraTreesRegressor(n_estimators=200, n_jobs=-1)),
            "GradBoost": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5)),
            "AdaBoost": MultiOutputRegressor(AdaBoostRegressor(n_estimators=200, learning_rate=0.5)),
            "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=250, max_depth=8, learning_rate=0.05, subsample=0.9, n_jobs=-1)),
            "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=64, verbose=-1)),
            "CatBoost": MultiOutputRegressor(CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8, verbose=False))
        }
        self.models_deep = {
            "GRU": GRUModel(),
            "LSTM": LSTMModel(),
            "BiLSTM": BiLSTMModel(),
            "CNN1D": CNN1DModel(),
            "CNNLSTM": CNNLSTMModel(),
            "Transformer": TransformerModel()
        }

    def train_classical(self):
        for n, m in self.models_classical.items():
            logger.info(f"[TRAIN] {n}")
            m.fit(self.Xtr, self.ytr)
            self.models[n] = m
            save_model(m, n, self.dataset_hash, window=self.window)

    def train_deep(self, epochs=25, batch=256):
        X_seq = self.Xtr.reshape(-1, self.window, 6)
        y = self.ytr
        for n, m in self.models_deep.items():
            logger.info(f"[TRAIN] {n}")
            Ds = tud.TensorDataset(torch.tensor(X_seq, dtype=torch.float32),
                                   torch.tensor(y, dtype=torch.float32))
            Dl = tud.DataLoader(Ds, batch_size=batch, shuffle=True)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            loss = nn.MSELoss()
            for e in range(epochs):
                total = 0
                for xb, yb in Dl:
                    opt.zero_grad()
                    l = loss(m(xb), yb)
                    l.backward(); opt.step()
                    total += l.item()
                sch.step()
                logger.info(f"{n} Epoch {e+1}/{epochs} Loss={total/len(Dl):.6f}")
            self.models[n] = m
            save_model(m, n, self.dataset_hash, window=self.window)

    def evaluate_all(self):
        res=[]
        for n,m in self.models.items():
            if hasattr(m,"predict"): yp=m.predict(self.Xte)
            else:
                with torch.no_grad():
                    X_seq=torch.tensor(self.Xte.reshape(-1,self.window,6),dtype=torch.float32)
                    yp=m(X_seq).numpy()
            mse=mean_squared_error(self.yte,yp)
            res.append((n,mse))
            logger.info(f"{n:15s} MSE={mse:.6f}")
        res.sort(key=lambda x:x[1])
        logger.info("===== VALIDATION MSE RANKING =====")
        for i,(n,mse) in enumerate(res,1): logger.info(f"{i:2d}. {n:15s} → {mse:.6f}")
        return res

    def get_model(self, name): return self.models.get(name)
