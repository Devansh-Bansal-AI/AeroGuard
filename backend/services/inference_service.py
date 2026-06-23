"""
AeroGuard — Inference Service

Handles model loading, initialization, and prediction serving.
Supports multiple inference backends:
  1. ONNX Runtime INT8 (preferred — fastest, smallest)
  2. ONNX Runtime FP32 (fallback)
  3. PyTorch CPU (fallback)
  4. Simulated mode (no models needed)

KEY: This service is called by EngineSimulator to produce REAL model predictions.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from collections import deque


# Sensor names matching C-MAPSS useful sensors (after dropping constants)
USEFUL_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13",
                  "s14", "s15", "s17", "s20", "s21"]

# Map dashboard sensor names to C-MAPSS column names
DASHBOARD_TO_CMAPSS = {
    "T2": "s2",  # These are approximate mappings for the synthetic data
    "T24": "s3",
    "T30": "s4",
    "T50": "s7",
    "P2": "s11",
    "P15": "s12",
    "P30": "s8",
    "Nf": "s9",
    "Nc": "s14",
    "Ps30": "s15",
    "phi": "s13",
    "NRf": "s17",
    "NRc": "s20",
    "BPR": "s21",
}

SEQUENCE_LENGTH = 50


class InferenceService:
    """
    ML model inference service for the backend.
    
    Manages:
    - Loading trained models (ONNX INT8/FP32, PyTorch)
    - Providing predictions for incoming sensor data
    - Feature engineering from raw sensors to model input
    - Graceful fallback to heuristic predictions if models aren't available
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.model_dir = os.path.join(project_root, "models")
        self.data_dir = os.path.join(project_root, "data", "processed")
        
        self.rul_model = None       # Can be ONNX session or PyTorch model
        self.anomaly_detector = None # AnomalyDetector (IsolationForest + LSTM Autoencoder)
        self.scaler = None
        self.metadata = None
        self.is_ready = False
        self.mode = "simulated"     # "onnx_int8", "onnx_fp32", "pytorch", "simulated"
        self._pytorch_explain_model = None  # Cached PyTorch model for explainability
        
        # Feature engineering config
        self.n_features = None
        self.feature_cols = None
        self.window_sizes = [5, 10, 20]
    
    async def initialize(self):
        """
        Initialize models. Try loading in priority order:
        1. ONNX INT8 (edge-optimized, fastest)
        2. ONNX FP32 (edge-ready)
        3. PyTorch BiLSTM (full precision)
        4. Simulated mode (no models needed)
        """
        print("  Loading ML models...")
        
        # Load metadata
        meta_path = os.path.join(self.model_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.n_features = self.metadata.get("n_features", 112)
        
        # Load feature columns from preprocessing metadata
        preprocess_meta_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(preprocess_meta_path):
            with open(preprocess_meta_path) as f:
                preprocess_meta = json.load(f)
            self.feature_cols = preprocess_meta.get("feature_cols", [])
            self.n_features = preprocess_meta.get("n_features", 112)
        
        # Try ONNX INT8 first (preferred for edge)
        int8_path = os.path.join(self.model_dir, "rul_bilstm_int8.onnx")
        if os.path.exists(int8_path):
            try:
                import onnxruntime as ort
                self.rul_model = ort.InferenceSession(
                    int8_path, providers=["CPUExecutionProvider"]
                )
                self.mode = "onnx_int8"
                size_mb = os.path.getsize(int8_path) / (1024 * 1024)
                print(f"  [OK] ONNX INT8 model loaded ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"  [WARN] ONNX INT8 loading failed: {e}")
        
        # Try ONNX FP32
        if self.mode == "simulated":
            fp32_path = os.path.join(self.model_dir, "rul_bilstm.onnx")
            if os.path.exists(fp32_path):
                try:
                    import onnxruntime as ort
                    self.rul_model = ort.InferenceSession(
                        fp32_path, providers=["CPUExecutionProvider"]
                    )
                    self.mode = "onnx_fp32"
                    size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
                    print(f"  [OK] ONNX FP32 model loaded ({size_mb:.2f} MB)")
                except Exception as e:
                    print(f"  [WARN] ONNX FP32 loading failed: {e}")
        
        # Try PyTorch model
        if self.mode == "simulated":
            pth_path = os.path.join(self.model_dir, "rul_bilstm.pth")
            if os.path.exists(pth_path):
                try:
                    import torch
                    sys_path_added = False
                    ml_dir = os.path.join(self.project_root, "ml")
                    import sys
                    if ml_dir not in sys.path:
                        sys.path.insert(0, ml_dir)
                        sys_path_added = True
                    
                    from train_lstm import BiLSTMAttentionRUL
                    
                    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
                    self.rul_model = BiLSTMAttentionRUL(
                        n_features=checkpoint['n_features'],
                        hidden_size=checkpoint['hidden_size'],
                        num_layers=checkpoint['num_layers'],
                        dropout=checkpoint['dropout'],
                    )
                    self.rul_model.load_state_dict(checkpoint['model_state_dict'])
                    self.rul_model.eval()
                    self.mode = "pytorch"
                    print(f"  [OK] PyTorch BiLSTM model loaded")
                except Exception as e:
                    print(f"  [WARN] PyTorch loading failed: {e}")
        
        # Simulated fallback
        if self.mode == "simulated":
            print("  [INFO] No trained models found -- using simulated predictions")
            print("  [INFO] Run 'python ml/train_lstm.py' then 'python ml/export_edge.py' to train")
        
        # Load scaler if available
        scaler_path = os.path.join(self.data_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            import joblib
            self.scaler = joblib.load(scaler_path)
            print("  [OK] Feature scaler loaded")
        
        # Load anomaly detection models
        anomaly_meta_path = os.path.join(self.model_dir, "anomaly_metadata.json")
        if os.path.exists(anomaly_meta_path):
            try:
                import sys as _sys
                ml_dir = os.path.join(self.project_root, "ml")
                if ml_dir not in _sys.path:
                    _sys.path.insert(0, ml_dir)
                from anomaly import AnomalyDetector
                
                self.anomaly_detector = AnomalyDetector(n_features=self.n_features or 112)
                self.anomaly_detector.load(self.model_dir)
                print(f"  [OK] Anomaly detection loaded (IF + LSTM Autoencoder)")
            except Exception as e:
                print(f"  [WARN] Anomaly model loading failed: {e}")
                self.anomaly_detector = None
        
        self.is_ready = True
        print(f"  [INFO] Inference mode: {self.mode}")
    
    def _engineer_features_from_history(self, sensor_history: list) -> np.ndarray:
        """
        Build feature-engineered sequence from raw sensor history.
        
        Takes the last SEQUENCE_LENGTH readings, computes rolling statistics,
        normalizes, and returns a model-ready input tensor.
        
        Args:
            sensor_history: List of dicts, each with sensor name -> value.
                           Must have at least SEQUENCE_LENGTH entries.
        
        Returns:
            np.ndarray — shape depends on model mode:
              - ONNX/PyTorch: (1, SEQUENCE_LENGTH, n_features)
              - XGBoost: (1, n_features)
        """
        if len(sensor_history) < SEQUENCE_LENGTH:
            return None
        
        # Take last SEQUENCE_LENGTH + max_window readings for rolling stats
        max_window = max(self.window_sizes)
        lookback = min(len(sensor_history), SEQUENCE_LENGTH + max_window)
        recent = sensor_history[-lookback:]
        
        # Map dashboard sensor names to ordered list
        sensor_names_ordered = list(DASHBOARD_TO_CMAPSS.keys())
        
        # Build raw sensor matrix (lookback, 14) and handle NaNs/missing values
        raw_matrix = np.zeros((len(recent), len(sensor_names_ordered)))
        for i, reading in enumerate(recent):
            for j, sensor in enumerate(sensor_names_ordered):
                val = reading.get(sensor)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    raw_matrix[i, j] = np.nan
                else:
                    raw_matrix[i, j] = val
                    
        # Robust forward-fill and mean imputation using Pandas
        df = pd.DataFrame(raw_matrix)
        df.ffill(inplace=True)
        df.fillna(df.mean(), inplace=True)
        df.fillna(0, inplace=True)  # Ultimate fallback
        raw_matrix = df.to_numpy()
        
        # Compute rolling features
        feature_list = []
        
        # Start with raw sensor values
        feature_list.append(raw_matrix)
        
        # Rolling mean and std for each window size
        for window in self.window_sizes:
            means = np.zeros_like(raw_matrix)
            stds = np.zeros_like(raw_matrix)
            for i in range(len(raw_matrix)):
                start = max(0, i - window + 1)
                window_data = raw_matrix[start:i+1]
                means[i] = window_data.mean(axis=0)
                stds[i] = window_data.std(axis=0) if len(window_data) > 1 else 0
            feature_list.append(means)
            feature_list.append(stds)
        
        # Rate of change (diff)
        diffs = np.zeros_like(raw_matrix)
        diffs[1:] = np.diff(raw_matrix, axis=0)
        feature_list.append(diffs)
        
        # Concatenate all features: 14 + (14*2*3) + 14 = 14 + 84 + 14 = 112
        all_features = np.concatenate(feature_list, axis=1)
        
        # Take last SEQUENCE_LENGTH timesteps
        sequence = all_features[-SEQUENCE_LENGTH:]
        
        # Normalize if scaler is available
        if self.scaler is not None:
            try:
                n_cols = min(sequence.shape[1], self.scaler.n_features_in_)
                sequence[:, :n_cols] = self.scaler.transform(sequence[:, :n_cols])
            except Exception:
                # If scaler mismatch, do simple min-max normalization
                mins = sequence.min(axis=0)
                maxs = sequence.max(axis=0)
                ranges = maxs - mins
                ranges[ranges == 0] = 1
                sequence = (sequence - mins) / ranges
        else:
            # Simple normalization without scaler
            mins = sequence.min(axis=0)
            maxs = sequence.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1
            sequence = (sequence - mins) / ranges
        
        # Return shape depends on model type
        if self.mode in ("onnx_int8", "onnx_fp32", "pytorch"):
            # Sequential models need full sequence: (1, seq_len, n_features)
            return sequence[np.newaxis, :].astype(np.float32)
        else:
            # XGBoost/tabular: last timestep only (1, n_features)
            return sequence[-1:].astype(np.float32)
    
    def predict_from_sensors(self, sensor_history: list, 
                              fallback_rul: float = 100.0) -> dict:
        """
        Predict RUL from raw sensor history. This is the main entry point
        called by EngineSimulator for REAL model inference.
        
        Args:
            sensor_history: List of dicts with sensor readings
            fallback_rul: Fallback RUL if model can't predict
        
        Returns:
            Dict with 'rul', 'anomaly_score', 'model_used'
        """
        if self.mode == "simulated" or len(sensor_history) < SEQUENCE_LENGTH:
            return {
                "rul": fallback_rul,
                "anomaly_score": None,  # Let simulator calculate
                "model_used": "simulated",
            }
        
        try:
            # Build feature-engineered input
            X = self._engineer_features_from_history(sensor_history)
            if X is None:
                return {"rul": fallback_rul, "anomaly_score": None, "model_used": "simulated"}
            
            # Run RUL model inference
            rul_pred = float(self.predict_rul(X).flatten()[0])
            
            # Clamp to reasonable range
            rul_pred = max(0.0, min(300.0, rul_pred))
            
            # Run anomaly detection if available
            anomaly_result = self.predict_anomaly(X)
            
            return {
                "rul": rul_pred,
                "anomaly_score": anomaly_result["combined_score"] if anomaly_result else None,
                "is_anomaly": anomaly_result["is_anomaly"] if anomaly_result else None,
                "model_used": self.mode,
            }
        except Exception as e:
            # Graceful fallback
            return {"rul": fallback_rul, "anomaly_score": None, "model_used": f"fallback ({e})"}
    
    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        """
        Predict Remaining Useful Life from pre-processed tensor.
        
        Args:
            X: Input array of shape (batch, seq_len, n_features)
        
        Returns:
            RUL predictions as numpy array
        """
        if self.mode in ("onnx_int8", "onnx_fp32"):
            input_name = self.rul_model.get_inputs()[0].name
            result = self.rul_model.run(None, {input_name: X})
            return result[0]
        
        elif self.mode == "pytorch":
            import torch
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                result = self.rul_model(X_tensor)
                return result.numpy()
        
        else:
            # Simulated prediction
            batch_size = X.shape[0] if X.ndim >= 2 else 1
            return np.array([100.0] * batch_size)
    
    def predict_anomaly(self, X: np.ndarray) -> dict:
        """
        Run anomaly detection on pre-processed tensor.
        
        Uses the dual-approach AnomalyDetector:
        - Isolation Forest on last-timestep features
        - LSTM Autoencoder reconstruction error
        - Combined weighted score
        
        Args:
            X: Input array of shape (batch, seq_len, n_features)
        
        Returns:
            Dict with 'combined_score', 'is_anomaly', 'ae_error', 'threshold'
            or None if anomaly model is not loaded
        """
        if self.anomaly_detector is None or not self.anomaly_detector.is_fitted:
            return None
        
        try:
            # AnomalyDetector.predict_single expects (seq_len, n_features) or (1, seq_len, n_features)
            result = self.anomaly_detector.predict_single(X[0] if X.ndim == 3 else X)
            return result
        except Exception as e:
            print(f"  [WARN] Anomaly prediction failed: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """Get model metadata and configuration."""
        info = {
            "mode": self.mode,
            "is_ready": self.is_ready,
        }
        
        if self.metadata:
            info.update({
                "model_type": self.metadata.get("model_type"),
                "n_features": self.metadata.get("n_features"),
                "sequence_length": self.metadata.get("sequence_length"),
                "hidden_size": self.metadata.get("hidden_size"),
                "total_params": self.metadata.get("total_params"),
                "rmse": self.metadata.get("rmse"),
                "mae": self.metadata.get("mae"),
                "nasa_score": self.metadata.get("nasa_score"),
            })
        
        # Model file sizes
        model_files = {
            "bilstm_pth": "rul_bilstm.pth",
            "onnx_fp32": "rul_bilstm.onnx",
            "onnx_int8": "rul_bilstm_int8.onnx",
        }
        for name, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                info[f"{name}_size_mb"] = round(size_mb, 3)
        
        # Load benchmark results if available
        bench_path = os.path.join(self.model_dir, "benchmark_results.json")
        if os.path.exists(bench_path):
            with open(bench_path) as f:
                info["benchmarks"] = json.load(f)
        
        return info
    
    def explain_prediction(self, sensor_history: list) -> dict:
        """
        Generate attention-based explainability for a prediction.
        
        Uses the PyTorch model's predict_with_attention() to extract
        temporal attention weights, showing which timesteps the model
        focused on most when predicting RUL.
        
        Args:
            sensor_history: List of dicts with sensor readings
        
        Returns:
            Dict with prediction, attention weights, and top contributing timesteps
            or None if explainability is not available
        """
        # Attention explainability requires PyTorch model (not ONNX)
        pth_path = os.path.join(self.model_dir, "rul_bilstm.pth")
        if not os.path.exists(pth_path) or len(sensor_history) < SEQUENCE_LENGTH:
            return None
        
        try:
            import torch
            import sys as _sys
            ml_dir = os.path.join(self.project_root, "ml")
            if ml_dir not in _sys.path:
                _sys.path.insert(0, ml_dir)
            from train_lstm import BiLSTMAttentionRUL
            
            # Use cached model if available, otherwise load once
            if self._pytorch_explain_model is None:
                checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
                model = BiLSTMAttentionRUL(
                    n_features=checkpoint['n_features'],
                    hidden_size=checkpoint['hidden_size'],
                    num_layers=checkpoint['num_layers'],
                    dropout=checkpoint['dropout'],
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self._pytorch_explain_model = model
            
            model = self._pytorch_explain_model
            
            # Build input tensor
            X = self._engineer_features_from_history(sensor_history)
            if X is None:
                return None
            
            X_tensor = torch.FloatTensor(X)
            
            # Get prediction with attention weights
            with torch.no_grad():
                rul_pred, attn_weights = model.predict_with_attention(X_tensor)
            
            rul = float(rul_pred.flatten()[0])
            weights = attn_weights.flatten().tolist()
            
            # Find top contributing timesteps
            top_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:10]
            
            return {
                "rul_prediction": round(rul, 2),
                "attention_weights": [round(w, 6) for w in weights],
                "top_timesteps": [
                    {"timestep": idx, "weight": round(weights[idx], 6)}
                    for idx in top_indices
                ],
                "seq_length": len(weights),
                "interpretation": (
                    "Higher attention weight = model focused more on that timestep. "
                    "Recent timesteps with high weights indicate acute degradation signals."
                ),
            }
        except Exception as e:
            print(f"  [WARN] Explainability failed: {e}")
            return None
