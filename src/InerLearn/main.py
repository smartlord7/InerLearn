"""
imu_pipeline.py
================
Public API for the complete IMU bias learning and evaluation pipeline.

This module integrates:
  • Environment & trajectory generation
  • IMU sensor data simulation
  • Supervised dataset preparation
  • Model training (classical + deep)
  • Bias correction evaluation & visualization

All steps use structured logging and can be invoked programmatically or as a standalone run.
"""

import logging
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

from environment import Environment, TrajectoryGenerator, plot_environment, load_or_create_environment
from imu_simulator import generate_imu_dataset, build_supervised_dataset
from model_provider import ModelProvider, load_model_cache, save_model_cache
from imu_evaluator import IMUEvaluator


# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logger = logging.getLogger("imu_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------------------------------------------------------
# Pipeline Functions
# -------------------------------------------------------------------
def generate_environment_and_trajectory(area_size=80, n_obstacles=10, T=120.0, dt=0.05, v_mean=2.0):
    """
    Generate the physical simulation environment and a realistic robot trajectory.
    """
    logger.info("Step 1: Generating environment and trajectory ...")
    env = load_or_create_environment(
        path="cache/environment.pkl",
        area_size=area_size,
        n_obstacles=10,
        n_goals=12,
        seed=42
    )
    traj = TrajectoryGenerator(env, T=2000, dt=dt, v_mean=v_mean)
    t, p_true, v_true, a_true, gyro_true = traj.generate()
    logger.info(f"Trajectory generated ({len(t)} steps, duration={t[-1]:.1f}s)")
    return env, t, p_true, v_true, a_true, gyro_true


def simulate_imu_data(env, t, p_true, a_true, gyro_true, dt=0.05):
    """
    Generate IMU accelerometer and gyroscope measurements with noise and bias.
    """
    logger.info("Step 2: Simulating IMU sensor data ...")
    acc_meas, gyro_meas, acc_bias_true, gyro_bias_true = generate_imu_dataset(env, t, p_true, a_true, gyro_true, dt)
    logger.info(f"IMU dataset: acc={acc_meas.shape}, gyro={gyro_meas.shape}")
    return acc_meas, gyro_meas, acc_bias_true, gyro_bias_true


def prepare_datasets(acc_meas, gyro_meas, acc_bias_true, window=40, split_ratio=0.7):
    """
    Prepare supervised learning datasets for IMU bias prediction.
    """
    logger.info("Step 3: Preparing supervised datasets ...")
    X, y = build_supervised_dataset(acc_meas, gyro_meas, acc_bias_true, window=window)
    split = int(split_ratio * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    logger.info(f"Dataset ready: {len(Xtr_s)} train, {len(Xte_s)} test samples")
    return Xtr_s, Xte_s, ytr, yte, scaler


def train_models(Xtr_s, ytr, Xte_s, yte, window=40, epochs=20):
    """
    Train all classical and deep estimators using the ModelProvider.
    """
    logger.info("Step 4: Training model suite ...")
    provider = load_model_cache()
    if provider is None:
        provider = ModelProvider(Xtr_s, ytr, Xte_s, yte, window=40)
        provider.train_classical()
        provider.train_deep(epochs=30)
        save_model_cache(provider)
        provider.train_classical()
    provider.train_deep(epochs=epochs)
    results = provider.evaluate_all()
    best_model_name, best_mse = results[0]
    best_model = provider.get_model(best_model_name)
    logger.info(f"✅ Best model: {best_model_name} (MSE={best_mse:.6f})")
    return provider, best_model_name, best_model


def evaluate_bias_correction(acc_meas, gyro_meas, a_true, t, env, scaler, estimators, window=40, dt=0.05):
    """
    Evaluate the learned models by correcting IMU bias and visualizing trajectories.
    """
    logger.info("Step 5: Evaluating bias correction performance ...")
    evaluator = IMUEvaluator(acc_meas, gyro_meas, a_true, t, env, scaler, estimators, window=window, dt=dt)
    summary = evaluator.evaluate()
    evaluator.plot_error_curves()
    evaluator.plot_trajectories()
    evaluator.animate_best()
    logger.info("Bias correction evaluation complete.")
    return summary


# -------------------------------------------------------------------
# Master Pipeline Function
# -------------------------------------------------------------------
def run_full_pipeline(
    area_size=5000,
    n_obstacles=500,
    T=120.0,
    dt=0.05,
    v_mean=2.2,
    window=40,
    epochs=20,
    plot_trajectory=True
):
    """
    Execute the full IMU bias learning pipeline.

    Parameters
    ----------
    area_size : float
        Size of the environment in meters.
    n_obstacles : int
        Number of obstacles to place.
    T : float
        Duration of the simulated run (s).
    dt : float
        Time step (s).
    v_mean : float
        Mean robot velocity (m/s).
    window : int
        Temporal window for supervised bias prediction.
    epochs : int
        Number of epochs for deep model training.
    plot_trajectory : bool
        Whether to show the 3D environment visualization.
    """
    env, t, p_true, v_true, a_true, gyro_true = generate_environment_and_trajectory(
        area_size, n_obstacles, T, dt, v_mean
    )
    if plot_trajectory:
        plot_environment(env, p_true, t)

    acc_meas, gyro_meas, acc_bias_true, gyro_bias_true = simulate_imu_data(env, t, p_true, a_true, gyro_true, dt)
    Xtr_s, Xte_s, ytr, yte, scaler = prepare_datasets(acc_meas, gyro_meas, acc_bias_true, window)
    provider, best_name, best_model = train_models(Xtr_s, ytr, Xte_s, yte, window, epochs)
    summary = evaluate_bias_correction(acc_meas, gyro_meas, a_true, t, env, scaler, provider.models, window, dt)

    logger.info("Pipeline execution completed successfully ✅")
    return summary, provider, best_model


# -------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    summary, provider, best_model = run_full_pipeline()
