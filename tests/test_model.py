"""Test model quality and ONNX export integrity."""
import os
import sys
import json
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def test_model_metadata_exists():
    """Verify model metadata file exists and has required fields."""
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    assert os.path.exists(meta_path), "model_metadata.json not found"
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    assert "rmse" in meta
    assert "mae" in meta
    assert "nasa_score" in meta
    assert "n_features" in meta
    assert meta["model_type"] == "bilstm_attention"


def test_model_rmse_within_threshold():
    """Verify RMSE is within acceptable range for synthetic data."""
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    
    # On synthetic data, RMSE ~44 is the ceiling. 
    # On real C-MAPSS, target would be < 20.
    assert meta["rmse"] < 50, f"RMSE {meta['rmse']} exceeds threshold of 50"


def test_onnx_models_exist():
    """Verify both ONNX FP32 and INT8 models are saved."""
    fp32_path = os.path.join(MODEL_DIR, "rul_bilstm.onnx")
    int8_path = os.path.join(MODEL_DIR, "rul_bilstm_int8.onnx")
    
    assert os.path.exists(fp32_path), "ONNX FP32 model not found"
    assert os.path.exists(int8_path), "ONNX INT8 model not found"


def test_onnx_int8_smaller_than_fp32():
    """Verify INT8 quantization actually compressed the model."""
    fp32_size = os.path.getsize(os.path.join(MODEL_DIR, "rul_bilstm.onnx"))
    int8_size = os.path.getsize(os.path.join(MODEL_DIR, "rul_bilstm_int8.onnx"))
    
    assert int8_size < fp32_size, "INT8 model should be smaller than FP32"
    compression = (1 - int8_size / fp32_size) * 100
    assert compression > 30, f"Compression {compression:.1f}% is too low (expected > 30%)"


def test_onnx_int8_predictions_close_to_pytorch():
    """Verify INT8 predictions match PyTorch within acceptable tolerance."""
    import torch
    import onnxruntime as ort
    
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))
    from train_lstm import BiLSTMAttentionRUL
    
    # Load PyTorch model
    pth_path = os.path.join(MODEL_DIR, "rul_bilstm.pth")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    model = BiLSTMAttentionRUL(
        n_features=checkpoint['n_features'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load ONNX INT8 model
    int8_path = os.path.join(MODEL_DIR, "rul_bilstm_int8.onnx")
    session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    
    # Generate test input
    test_input = torch.randn(3, checkpoint['seq_len'], checkpoint['n_features'])
    
    # PyTorch prediction
    with torch.no_grad():
        pytorch_out = model(test_input).numpy().flatten()
    
    # ONNX INT8 prediction
    int8_out = session.run(None, {"sensor_sequence": test_input.numpy()})[0].flatten()
    
    # Check closeness (INT8 can have up to ~2 cycles of error)
    max_diff = np.max(np.abs(pytorch_out - int8_out))
    assert max_diff < 2.0, f"INT8 diverges from PyTorch by {max_diff:.4f} cycles (max allowed: 2.0)"


def test_benchmark_results_exist():
    """Verify benchmark results have been generated."""
    bench_path = os.path.join(MODEL_DIR, "benchmark_results.json")
    assert os.path.exists(bench_path), "benchmark_results.json not found"
    
    with open(bench_path) as f:
        results = json.load(f)
    
    assert "pytorch_cpu" in results
    assert "onnx_fp32" in results
    assert "onnx_int8" in results
    assert results["speedup_int8_vs_pytorch"] > 1.0, "INT8 should be faster than PyTorch"
