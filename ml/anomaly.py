"""
AeroGuard — Anomaly Detection Module

Dual-approach anomaly detection for aircraft engine health:

1. Isolation Forest: Statistical anomaly detection on sensor features
   - Fast, interpretable, works well with tabular data
   - Good for catching sudden out-of-distribution readings

2. LSTM Autoencoder: Learned reconstruction-based anomaly detection
   - Trains on healthy engine data to learn "normal" patterns
   - High reconstruction error = anomaly
   - Better at catching subtle degradation patterns

Both approaches feed into a combined anomaly score.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
import joblib
import os
import json


# ─── LSTM Autoencoder ──────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for sequence reconstruction-based anomaly detection.
    
    Encoder compresses the sequence into a latent vector.
    Decoder reconstructs the original sequence.
    High reconstruction error indicates anomalous behavior.
    
    Args:
        n_features: Number of input features per timestep
        hidden_size: LSTM hidden dimension
        latent_dim: Dimension of bottleneck layer
    """
    
    def __init__(self, n_features: int, hidden_size: int = 64, latent_dim: int = 16):
        super(LSTMAutoencoder, self).__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True
        )
    
    def encode(self, x):
        """Encode sequence to latent representation."""
        lstm_out, (h_n, _) = self.encoder_lstm(x)
        # Use final hidden state as sequence representation
        latent = self.encoder_fc(h_n.squeeze(0))
        return latent
    
    def decode(self, latent, seq_len):
        """Decode latent representation back to sequence."""
        hidden = self.decoder_fc(latent)
        # Repeat latent for each timestep
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(hidden)
        return decoded
    
    def forward(self, x):
        """Full forward pass: encode then decode."""
        latent = self.encode(x)
        reconstructed = self.decode(latent, x.shape[1])
        return reconstructed
    
    def get_reconstruction_error(self, x):
        """Calculate per-sample mean reconstruction error."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error.cpu().numpy()


# ─── Combined Anomaly Detector ────────────────────────────────────────

class AnomalyDetector:
    """
    Combined anomaly detection using Isolation Forest + LSTM Autoencoder.
    
    The combined score is a weighted average of:
    - Isolation Forest anomaly score (normalized)
    - LSTM Autoencoder reconstruction error (normalized)
    
    Threshold is calibrated on healthy training data (99th percentile).
    """
    
    def __init__(self, n_features: int, contamination: float = 0.05):
        self.n_features = n_features
        self.contamination = contamination
        
        # Isolation Forest for statistical anomaly detection
        self.iforest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # LSTM Autoencoder for learned anomaly detection
        self.autoencoder = LSTMAutoencoder(n_features=n_features)
        
        # Thresholds (set during training)
        self.ae_threshold = None
        self.ae_mean = None
        self.ae_std = None
        self.is_fitted = False
    
    def fit_isolation_forest(self, X_flat):
        """
        Fit Isolation Forest on flattened features.
        
        Args:
            X_flat: 2D array of shape (n_samples, n_features)
        """
        print("  Training Isolation Forest...")
        self.iforest.fit(X_flat)
        print(f"  [OK] Isolation Forest fitted on {len(X_flat)} samples")
    
    def fit_autoencoder(self, X_sequences, epochs: int = 30, batch_size: int = 128,
                        lr: float = 0.001):
        """
        Train LSTM Autoencoder on healthy engine data.
        
        Args:
            X_sequences: 3D array (n_samples, seq_len, n_features), healthy data only
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        
        dataset = TensorDataset(torch.FloatTensor(X_sequences))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        print("  Training LSTM Autoencoder...")
        
        for epoch in range(epochs):
            self.autoencoder.train()
            losses = []
            
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} — Loss: {np.mean(losses):.6f}")
        
        # Calculate threshold on training data
        self.autoencoder.eval()
        with torch.no_grad():
            all_errors = []
            for (batch,) in loader:
                batch = batch.to(device)
                errors = self.autoencoder.get_reconstruction_error(batch)
                all_errors.extend(errors)
        
        all_errors = np.array(all_errors)
        self.ae_mean = float(np.mean(all_errors))
        self.ae_std = float(np.std(all_errors))
        self.ae_threshold = float(np.percentile(all_errors, 99))
        
        print(f"  [OK] Autoencoder trained. Threshold: {self.ae_threshold:.6f}")
        self.autoencoder.cpu()
    
    def fit(self, X_sequences, healthy_ratio: float = 0.6):
        """
        Fit both anomaly detection models.
        
        Uses the first `healthy_ratio` of each engine's data as "healthy" baseline.
        
        Args:
            X_sequences: 3D array (n_samples, seq_len, n_features)
            healthy_ratio: Fraction of data considered healthy
        """
        print("\n" + "=" * 50)
        print("AeroGuard Anomaly Detection Training")
        print("=" * 50)
        
        # Use first portion of data as "healthy" baseline
        n_healthy = int(len(X_sequences) * healthy_ratio)
        X_healthy = X_sequences[:n_healthy]
        
        # Flatten for Isolation Forest (use last timestep features)
        X_flat = X_sequences[:, -1, :]  # Shape: (n_samples, n_features)
        self.fit_isolation_forest(X_flat)
        
        # Train autoencoder on healthy data
        self.fit_autoencoder(X_healthy, epochs=30)
        
        self.is_fitted = True
        print("\n[OK] Anomaly detection models trained successfully")
    
    def predict(self, X_sequences):
        """
        Predict anomaly scores for input sequences.
        
        Args:
            X_sequences: 3D array (n_samples, seq_len, n_features)
        
        Returns:
            Dictionary with combined scores, IF scores, AE scores, and labels
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before prediction")
        
        # Isolation Forest scores
        X_flat = X_sequences[:, -1, :]
        if_scores = -self.iforest.score_samples(X_flat)  # Higher = more anomalous
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        
        # Autoencoder reconstruction error
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequences)
            ae_errors = self.autoencoder.get_reconstruction_error(X_tensor)
        
        ae_scores_norm = (ae_errors - self.ae_mean) / (self.ae_std + 1e-8)
        ae_scores_norm = np.clip(ae_scores_norm, 0, 10) / 10  # Normalize to [0, 1]
        
        # Combined score (weighted average)
        combined = 0.4 * if_scores_norm + 0.6 * ae_scores_norm
        
        # Binary labels
        is_anomaly = (ae_errors > self.ae_threshold) | (self.iforest.predict(X_flat) == -1)
        
        return {
            "combined_score": combined,
            "isolation_forest_score": if_scores_norm,
            "autoencoder_score": ae_scores_norm,
            "autoencoder_error": ae_errors,
            "is_anomaly": is_anomaly.astype(int),
            "threshold": self.ae_threshold,
        }
    
    def predict_single(self, X_single):
        """
        Predict anomaly for a single sequence.
        
        Args:
            X_single: 2D array (seq_len, n_features) or 3D (1, seq_len, n_features)
        
        Returns:
            Dictionary with anomaly score and label
        """
        if X_single.ndim == 2:
            X_single = X_single[np.newaxis, :]
        
        result = self.predict(X_single)
        
        return {
            "combined_score": float(result["combined_score"][0]),
            "is_anomaly": bool(result["is_anomaly"][0]),
            "autoencoder_error": float(result["autoencoder_error"][0]),
            "threshold": self.ae_threshold,
        }
    
    def save(self, save_dir: str):
        """Save both models and metadata."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Isolation Forest
        joblib.dump(self.iforest, os.path.join(save_dir, "isolation_forest.joblib"))
        
        # Save Autoencoder
        torch.save(
            self.autoencoder.state_dict(),
            os.path.join(save_dir, "autoencoder.pth")
        )
        
        # Save metadata
        meta = {
            "n_features": self.n_features,
            "contamination": self.contamination,
            "ae_threshold": self.ae_threshold,
            "ae_mean": self.ae_mean,
            "ae_std": self.ae_std,
        }
        with open(os.path.join(save_dir, "anomaly_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"Anomaly detector saved to {save_dir}")
    
    def load(self, save_dir: str):
        """Load both models and metadata."""
        # Load Isolation Forest
        self.iforest = joblib.load(os.path.join(save_dir, "isolation_forest.joblib"))
        
        # Load Autoencoder
        self.autoencoder = LSTMAutoencoder(n_features=self.n_features)
        self.autoencoder.load_state_dict(
            torch.load(os.path.join(save_dir, "autoencoder.pth"), map_location="cpu")
        )
        self.autoencoder.eval()
        
        # Load metadata
        with open(os.path.join(save_dir, "anomaly_metadata.json")) as f:
            meta = json.load(f)
        
        self.ae_threshold = meta["ae_threshold"]
        self.ae_mean = meta["ae_mean"]
        self.ae_std = meta["ae_std"]
        self.is_fitted = True
        
        print(f"Anomaly detector loaded from {save_dir}")


# ─── Main Training Script ─────────────────────────────────────────────

if __name__ == "__main__":
    from preprocess import preprocess_pipeline
    
    # Preprocess data
    data = preprocess_pipeline(use_synthetic=True)
    
    X_train = data["X_train"]
    n_features = X_train.shape[2]
    
    # Train anomaly detector
    detector = AnomalyDetector(n_features=n_features, contamination=0.05)
    detector.fit(X_train, healthy_ratio=0.6)
    
    # Test predictions
    X_test = data["X_test"]
    results = detector.predict(X_test)
    
    n_anomalies = results["is_anomaly"].sum()
    print(f"\nTest results: {n_anomalies}/{len(X_test)} anomalies detected "
          f"({100*n_anomalies/len(X_test):.1f}%)")
    
    # Save
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    detector.save(model_dir)
    
    print("\n[OK] Anomaly detection training complete!")
