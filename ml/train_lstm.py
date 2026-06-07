"""
AeroGuard — BiLSTM + Attention RUL Predictor

Bidirectional LSTM with temporal attention for Remaining Useful Life prediction
on NASA C-MAPSS turbofan engine degradation data.

Architecture:
    Input → BiLSTM(2 layers) → Temporal Attention → FC → RUL

Key design decisions:
    - Bidirectional LSTM captures both forward and backward temporal patterns
    - Temporal attention learns which timesteps matter most for RUL prediction
    - Huber loss is used (less sensitive to outliers than MSE, better NASA scores)
    - Learning rate scheduling with ReduceLROnPlateau for stable convergence
    - Piecewise linear RUL target (capped at 125) per NASA convention
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────

PROJECT_ROOT = str(Path(__file__).parent.parent)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Hyperparameters
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.5
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
EPOCHS = 80
PATIENCE = 18          # Early stopping patience
MAX_RUL = 125          # Matches preprocessing


# ─── Model Architecture ───────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism that learns to weight each timestep.
    
    Given BiLSTM output (batch, seq_len, hidden*2), produces a weighted
    context vector by learning which timesteps are most informative
    for predicting RUL.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden*2)
        scores = self.attn(lstm_output)          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)   # (batch, seq_len, 1)
        context = (lstm_output * weights).sum(dim=1)  # (batch, hidden*2)
        return context, weights.squeeze(-1)


class BiLSTMAttentionRUL(nn.Module):
    """
    Bidirectional LSTM with Temporal Attention for RUL prediction.
    
    Args:
        n_features: Number of input features per timestep
        hidden_size: LSTM hidden dimension (each direction)
        num_layers: Number of stacked LSTM layers
        dropout: Dropout rate between LSTM layers and in FC
    """
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection (stabilizes training with wide feature space)
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Temporal attention over BiLSTM outputs
        self.attention = TemporalAttention(hidden_size * 2)
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            rul: (batch, 1) — predicted RUL
        """
        # Project input features
        x = self.input_proj(x)           # (batch, seq_len, hidden_size)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden_size*2)
        
        # Temporal attention
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden*2)
        
        # Predict RUL
        rul = self.fc(context)           # (batch, 1)
        return rul
    
    def predict_with_attention(self, x):
        """Return both prediction and attention weights for explainability."""
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        rul = self.fc(context)
        return rul, attn_weights


# ─── NASA Scoring Function ────────────────────────────────────────────

def nasa_scoring_function(y_true, y_pred):
    """
    Asymmetric scoring from NASA's C-MAPSS challenge.
    Late predictions (predicting failure after it happens) are penalized
    more heavily than early predictions.
    
    s = sum( exp(-d/13) - 1 )  for d < 0 (early)
    s = sum( exp( d/10) - 1 )  for d >= 0 (late)
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(score))


# ─── Training Pipeline ────────────────────────────────────────────────

def train_bilstm():
    print("=" * 60)
    print("AeroGuard BiLSTM+Attention RUL Training Pipeline")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # ── 1. Load Data ──────────────────────────────────────────────
    print("\n[1/5] Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        data_meta = json.load(f)
    
    n_features = X_train.shape[2]
    seq_len = X_train.shape[1]
    
    print(f"  X_train: {X_train.shape}  (samples, seq_len, features)")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"  Features: {n_features}, Sequence Length: {seq_len}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=device.type == "cuda")
    
    # ── 2. Build Model ────────────────────────────────────────────
    print(f"\n[2/5] Building BiLSTM+Attention model...")
    model = BiLSTMAttentionRUL(
        n_features=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture: Input({n_features}) -> Proj({HIDDEN_SIZE}) -> "
          f"BiLSTM({HIDDEN_SIZE}x2, {NUM_LAYERS}L) -> Attention -> FC -> RUL")
    
    # ── 3. Training Setup ─────────────────────────────────────────
    print(f"\n[3/5] Training setup...")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"  Loss: HuberLoss (delta=1.0), Optimizer: AdamW")
    print(f"  Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    print(f"  Early stopping: patience={PATIENCE}")
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
    )
    
    # ── 4. Training Loop ──────────────────────────────────────────
    print(f"\n[4/5] Training...")
    print("-" * 70)
    print(f"{'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | {'Val RMSE':>9} | {'LR':>10} | {'Status'}")
    print("-" * 70)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # ── Validate ──
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())
                val_preds.append(y_pred.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())
        
        val_loss = np.mean(val_losses)
        val_preds_arr = np.concatenate(val_preds).flatten()
        val_targets_arr = np.concatenate(val_targets).flatten()
        val_rmse = np.sqrt(mean_squared_error(val_targets_arr, val_preds_arr))
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # ── Early Stopping Check ──
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status = "* BEST"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                status = "EARLY STOP"
            elif patience_counter >= PATIENCE - 3:
                status = f"! patience {patience_counter}/{PATIENCE}"
        
        if epoch % 3 == 0 or epoch == 1 or status:
            print(f"{epoch:>5} | {train_loss:>11.4f} | {val_loss:>11.4f} | {val_rmse:>9.2f} | {current_lr:>10.6f} | {status}")
        
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    train_time = time.time() - train_start
    print("-" * 70)
    print(f"  Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")
    
    # ── 5. Final Evaluation ───────────────────────────────────────
    print(f"\n[5/5] Final Evaluation on Test Set...")
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()
    
    # Clamp predictions to valid range
    y_pred = np.clip(y_pred, 0, MAX_RUL)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nasa_score = nasa_scoring_function(y_true, y_pred)
    
    print("=" * 40)
    print("  BiLSTM+Attention Test Metrics")
    print("=" * 40)
    print(f"  RMSE:        {rmse:.2f} cycles")
    print(f"  MAE:         {mae:.2f} cycles")
    print(f"  NASA Score:  {nasa_score:.2f}")
    print(f"  Parameters:  {total_params:,}")
    print("=" * 40)
    
    # ── Save Model ────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save PyTorch model
    model_path = os.path.join(MODEL_DIR, "rul_bilstm.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'n_features': n_features,
        'seq_len': seq_len,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
    }, model_path)
    
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n  Model saved: {model_path} ({model_size_mb:.2f} MB)")
    
    # Save metadata
    metadata = {
        "model_type": "bilstm_attention",
        "n_features": n_features,
        "sequence_length": seq_len,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "total_params": total_params,
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "nasa_score": round(float(nasa_score), 4),
        "model_size_mb": round(model_size_mb, 3),
        "training_time_s": round(train_time, 1),
        "epochs_trained": epoch,
        "device": str(device),
    }
    
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved: {meta_path}")
    print("\n" + "=" * 60)
    print("  [OK] BiLSTM+Attention training complete!")
    print("=" * 60)
    
    return model, metadata


if __name__ == "__main__":
    model, meta = train_bilstm()
