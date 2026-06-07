"""
AeroGuard — NASA C-MAPSS Data Preprocessor & Synthetic Generator

This module handles:
1. Loading NASA C-MAPSS turbofan engine degradation datasets
2. Feature engineering (rolling statistics, normalization, RUL labeling)
3. Generating realistic synthetic C-MAPSS data for instant demo capability
4. Preparing sequences for LSTM training

NASA C-MAPSS sensor columns:
    - Operational settings: altitude, Mach, TRA (3 cols)
    - Sensor measurements: T2, T24, T30, T50, P2, P15, P30, Nf, Nc,
      epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd,
      W31, W32 (21 cols)
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Configuration ---
SEQUENCE_LENGTH = 50  # Lookback window for LSTM
MAX_RUL = 125         # Cap RUL at 125 cycles (piecewise linear)
N_SENSORS = 14        # Number of useful sensors (after dropping constant ones)

SENSOR_NAMES = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
    "s20", "s21"
]

SETTING_NAMES = ["setting1", "setting2", "setting3"]

COLUMN_NAMES = ["engine_id", "cycle"] + SETTING_NAMES + SENSOR_NAMES

# Sensors that are approximately constant and provide no information
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Useful sensors for modeling
USEFUL_SENSORS = [s for s in SENSOR_NAMES if s not in DROP_SENSORS]


def load_cmapss(data_dir: str, dataset: str = "FD001"):
    """
    Load NASA C-MAPSS dataset files.
    
    Args:
        data_dir: Path to directory containing train_FD00X.txt and test_FD00X.txt
        dataset: One of FD001, FD002, FD003, FD004
    
    Returns:
        train_df, test_df, rul_df
    """
    train_path = os.path.join(data_dir, f"train_{dataset}.txt")
    test_path = os.path.join(data_dir, f"test_{dataset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{dataset}.txt")
    
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])
    
    return train_df, test_df, rul_df


def generate_synthetic_cmapss(n_engines: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic C-MAPSS-like data for demo purposes.
    
    Simulates turbofan engine degradation with:
    - Gradual sensor drift as engines degrade
    - Realistic noise levels per sensor
    - Engine-specific lifespans (128-362 cycles, matching real C-MAPSS distribution)
    - 3 operational settings + 21 sensor readings
    
    Args:
        n_engines: Number of engines to simulate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame in C-MAPSS format with columns matching COLUMN_NAMES
    """
    np.random.seed(seed)
    
    # Realistic sensor baselines and degradation profiles (inspired by FD001)
    sensor_profiles = {
        "s1":  {"base": 518.67, "drift": 0.0,    "noise": 0.0},
        "s2":  {"base": 642.44, "drift": 0.8,    "noise": 0.5},
        "s3":  {"base": 1589.7, "drift": 2.5,    "noise": 1.2},
        "s4":  {"base": 1400.6, "drift": 6.0,    "noise": 3.0},
        "s5":  {"base": 14.62,  "drift": 0.0,    "noise": 0.0},
        "s6":  {"base": 21.61,  "drift": 0.0,    "noise": 0.0},
        "s7":  {"base": 554.36, "drift": 1.5,    "noise": 0.8},
        "s8":  {"base": 2388.0, "drift": 0.5,    "noise": 0.2},
        "s9":  {"base": 9046.2, "drift": 3.0,    "noise": 2.0},
        "s10": {"base": 1.3,    "drift": 0.0,    "noise": 0.0},
        "s11": {"base": 47.47,  "drift": 0.15,   "noise": 0.05},
        "s12": {"base": 521.66, "drift": 0.7,    "noise": 0.4},
        "s13": {"base": 2388.0, "drift": 0.5,    "noise": 0.2},
        "s14": {"base": 8138.6, "drift": 2.5,    "noise": 1.5},
        "s15": {"base": 8.449,  "drift": 0.02,   "noise": 0.005},
        "s16": {"base": 0.03,   "drift": 0.0,    "noise": 0.0},
        "s17": {"base": 392.0,  "drift": 0.5,    "noise": 0.3},
        "s18": {"base": 2388.0, "drift": 0.0,    "noise": 0.0},
        "s19": {"base": 100.0,  "drift": 0.0,    "noise": 0.0},
        "s20": {"base": 39.06,  "drift": 0.1,    "noise": 0.03},
        "s21": {"base": 23.42,  "drift": 0.15,   "noise": 0.05},
    }
    
    all_rows = []
    
    for engine_id in range(1, n_engines + 1):
        # Random engine lifespan matching C-MAPSS FD001 distribution
        max_cycles = np.random.randint(128, 362)
        
        for cycle in range(1, max_cycles + 1):
            row = {"engine_id": engine_id, "cycle": cycle}
            
            # Operational settings
            row["setting1"] = np.random.choice([-0.0015, -0.0007, 0.0, 0.0007, 0.0015, 0.0025])
            row["setting2"] = np.random.choice([-0.0002, 0.0, 0.0002, 0.0003, 0.0005])
            row["setting3"] = 100.0
            
            # Degradation progress (0 to 1)
            progress = cycle / max_cycles
            
            # Non-linear degradation (accelerating towards end of life)
            degradation = progress ** 2.3
            
            # Add health-dependent acceleration in final 20% of life
            if progress > 0.8:
                degradation += (progress - 0.8) * 2.5
            
            for sensor_name, profile in sensor_profiles.items():
                base = profile["base"]
                drift = profile["drift"]
                noise = profile["noise"]
                
                # Sensor value = base + degradation_drift + noise
                value = base + drift * degradation * max_cycles * 0.01
                value += np.random.normal(0, noise)
                
                # Add occasional spikes in final 10% of life
                if progress > 0.9 and np.random.random() < 0.05:
                    value += drift * 3 * np.random.random()
                
                row[sensor_name] = round(value, 4)
            
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows, columns=COLUMN_NAMES)
    return df


def add_rul_labels(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """
    Add Remaining Useful Life labels to training data.
    
    Uses piecewise linear degradation:
    - RUL decreases linearly from max_rul to 0
    - Capped at max_rul for early cycles (constant health region)
    
    Args:
        df: DataFrame with engine_id and cycle columns
        max_rul: Maximum RUL cap (default 125)
    
    Returns:
        DataFrame with added 'rul' column
    """
    df = df.copy()
    
    # Calculate max cycle per engine
    max_cycles = df.groupby("engine_id")["cycle"].max()
    
    # Merge max cycle and calculate RUL
    df = df.merge(max_cycles.rename("max_cycle"), on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    
    # Apply piecewise linear cap
    df["rul"] = df["rul"].clip(upper=max_rul)
    
    df.drop("max_cycle", axis=1, inplace=True)
    return df


def engineer_features(df: pd.DataFrame, window_sizes: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Engineer rolling statistics features for each useful sensor.
    
    Creates per-engine rolling:
    - Mean
    - Standard deviation
    - Rate of change (diff)
    
    Args:
        df: DataFrame with sensor columns
        window_sizes: List of rolling window sizes
    
    Returns:
        DataFrame with added feature columns
    """
    df = df.copy()
    
    feature_cols = []
    
    for sensor in USEFUL_SENSORS:
        for window in window_sizes:
            # Rolling mean
            col_mean = f"{sensor}_rmean_{window}"
            df[col_mean] = df.groupby("engine_id")[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            feature_cols.append(col_mean)
            
            # Rolling std
            col_std = f"{sensor}_rstd_{window}"
            df[col_std] = df.groupby("engine_id")[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
            )
            feature_cols.append(col_std)
        
        # Rate of change
        col_diff = f"{sensor}_diff"
        df[col_diff] = df.groupby("engine_id")[sensor].diff().fillna(0)
        feature_cols.append(col_diff)
    
    return df, feature_cols


def normalize_features(df: pd.DataFrame, feature_cols: list, 
                       scaler: MinMaxScaler = None, fit: bool = True):
    """
    Normalize feature columns using MinMaxScaler.
    
    Args:
        df: DataFrame with feature columns
        feature_cols: List of column names to normalize
        scaler: Pre-fitted scaler (for test data)
        fit: Whether to fit the scaler
    
    Returns:
        DataFrame with normalized features, fitted scaler
    """
    df = df.copy()
    
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    
    cols_to_normalize = USEFUL_SENSORS + feature_cols
    
    if fit:
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    else:
        df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])
    
    return df, scaler


def create_sequences(df: pd.DataFrame, feature_cols: list, 
                     sequence_length: int = SEQUENCE_LENGTH):
    """
    Create time-series sequences for LSTM input.
    
    For each engine, creates sliding windows of (sequence_length) timesteps.
    Each sequence maps to the RUL at the last timestep.
    
    Args:
        df: Preprocessed DataFrame with features and RUL
        feature_cols: List of feature column names
        sequence_length: Number of timesteps per sequence
    
    Returns:
        X: np.array of shape (n_samples, sequence_length, n_features)
        y: np.array of shape (n_samples,) — RUL targets
        engine_ids: np.array of engine IDs for each sequence
    """
    all_features = USEFUL_SENSORS + feature_cols
    
    X_list, y_list, id_list = [], [], []
    
    for engine_id, group in df.groupby("engine_id"):
        data = group[all_features].values
        rul = group["rul"].values
        
        for i in range(len(data) - sequence_length + 1):
            X_list.append(data[i:i + sequence_length])
            y_list.append(rul[i + sequence_length - 1])
            id_list.append(engine_id)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    engine_ids = np.array(id_list)
    
    return X, y, engine_ids


def preprocess_pipeline(use_synthetic: bool = True, data_dir: str = None,
                       dataset: str = "FD001", save_dir: str = None):
    """
    Complete preprocessing pipeline from raw data to LSTM-ready sequences.
    
    Args:
        use_synthetic: If True, generate synthetic data (no download needed)
        data_dir: Path to C-MAPSS raw files (required if use_synthetic=False)
        dataset: C-MAPSS dataset name (FD001-FD004)
        save_dir: Directory to save processed data and scaler
    
    Returns:
        Dictionary with X_train, y_train, X_test, y_test, scaler, metadata
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("AeroGuard Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    if use_synthetic:
        print("\n[1/6] Generating synthetic C-MAPSS data (100 engines)...")
        train_df = generate_synthetic_cmapss(n_engines=80, seed=42)
        test_df = generate_synthetic_cmapss(n_engines=20, seed=99)
        
        # Generate test RUL (last cycle RUL for each engine)
        test_max = test_df.groupby("engine_id")["cycle"].max()
        # Simulate truncated test data (cut at random point)
        truncated_test = []
        test_ruls = []
        for eid, group in test_df.groupby("engine_id"):
            max_c = group["cycle"].max()
            cut_point = np.random.randint(int(max_c * 0.4), int(max_c * 0.85))
            truncated_test.append(group[group["cycle"] <= cut_point])
            test_ruls.append(max_c - cut_point)
        test_df = pd.concat(truncated_test, ignore_index=True)
        rul_test = np.array(test_ruls)
    else:
        print(f"\n[1/6] Loading NASA C-MAPSS {dataset}...")
        train_df, test_df, rul_df = load_cmapss(data_dir, dataset)
        rul_test = rul_df["rul"].values
    
    print(f"  Training: {len(train_df)} rows, {train_df['engine_id'].nunique()} engines")
    print(f"  Testing:  {len(test_df)} rows, {test_df['engine_id'].nunique()} engines")
    
    # Step 2: Add RUL labels
    print("\n[2/6] Adding RUL labels (piecewise linear, cap={})...".format(MAX_RUL))
    train_df = add_rul_labels(train_df)
    test_df = add_rul_labels(test_df)
    
    # Step 3: Feature engineering
    print("\n[3/6] Engineering rolling features...")
    train_df, feature_cols = engineer_features(train_df)
    test_df, _ = engineer_features(test_df)
    print(f"  Created {len(feature_cols)} engineered features")
    print(f"  Total features: {len(USEFUL_SENSORS) + len(feature_cols)}")
    
    # Step 4: Normalize
    print("\n[4/6] Normalizing features (MinMaxScaler)...")
    train_df, scaler = normalize_features(train_df, feature_cols, fit=True)
    test_df, _ = normalize_features(test_df, feature_cols, scaler=scaler, fit=False)
    
    # Step 5: Create sequences
    print(f"\n[5/6] Creating sequences (window={SEQUENCE_LENGTH})...")
    X_train, y_train, train_ids = create_sequences(train_df, feature_cols)
    X_test, y_test, test_ids = create_sequences(test_df, feature_cols)
    print(f"  Training sequences: {X_train.shape}")
    print(f"  Test sequences:     {X_test.shape}")
    
    # Step 6: Save
    print("\n[6/6] Saving processed data...")
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    np.save(os.path.join(save_dir, "train_ids.npy"), train_ids)
    np.save(os.path.join(save_dir, "test_ids.npy"), test_ids)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    
    metadata = {
        "dataset": dataset if not use_synthetic else "synthetic",
        "n_train_engines": int(train_df["engine_id"].nunique()),
        "n_test_engines": int(test_df["engine_id"].nunique()),
        "sequence_length": SEQUENCE_LENGTH,
        "n_features": X_train.shape[2],
        "n_train_sequences": int(X_train.shape[0]),
        "n_test_sequences": int(X_test.shape[0]),
        "max_rul": MAX_RUL,
        "useful_sensors": USEFUL_SENSORS,
        "feature_cols": feature_cols,
    }
    
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Data saved to {save_dir}")
    print("=" * 60)
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "metadata": metadata,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    result = preprocess_pipeline(use_synthetic=True)
    print(f"\nPreprocessing complete!")
    print(f"Training shape: {result['X_train'].shape}")
    print(f"Test shape: {result['X_test'].shape}")
