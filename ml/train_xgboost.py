"""
AeroGuard — XGBoost RUL Predictor Training

Replaces the overengineered BiLSTM with a fast, edge-optimized XGBoost model.
Uses rolling window features from the last timestep of the sequence to capture temporal dynamics.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = str(Path(__file__).parent.parent)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def evaluate_nasa_score(y_true, y_pred):
    """
    Standard asymmetric scoring metric for C-MAPSS.
    Late predictions are penalized more heavily than early predictions.
    """
    errors = y_pred - y_true
    score = 0.0
    for e in errors:
        if e < 0:
            score += np.exp(-e / 13.0) - 1.0
        else:
            score += np.exp(e / 10.0) - 1.0
    return score


def train_xgboost():
    print("=" * 60)
    print("AeroGuard XGBoost Edge Training Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/4] Loading preprocessed data...")
    X_train_3d = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test_3d = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        data_meta = json.load(f)
    
    # 2. Reshape for XGBoost (extract last timestep containing rolling features)
    print(f"\n[2/4] Flattening sequences for XGBoost...")
    X_train = X_train_3d[:, -1, :]
    X_test = X_test_3d[:, -1, :]
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    
    # 3. Train Model
    print("\n[3/4] Training XGBoost RUL Predictor...")
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=15
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=10
    )
    
    # 4. Evaluate
    print("\n[4/4] Evaluating Edge Performance...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    nasa_score = evaluate_nasa_score(y_test, y_pred)
    
    print("-" * 30)
    print("XGBoost Test Metrics:")
    print(f"  RMSE:       {rmse:.2f}")
    print(f"  MAE:        {mae:.2f}")
    print(f"  NASA Score: {nasa_score:.2f}")
    print("-" * 30)
    
    # Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "rul_model.json")
    model.save_model(model_path)
    
    # Save Metadata
    metadata = {
        "model_type": "xgboost",
        "n_features": X_train.shape[1],
        "rmse": float(rmse),
        "mae": float(mae),
        "nasa_score": float(nasa_score)
    }
    
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Model saved to {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_xgboost()
