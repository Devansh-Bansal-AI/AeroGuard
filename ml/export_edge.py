"""
AeroGuard — Edge Export Pipeline

Exports trained PyTorch models to ONNX format and applies INT8 quantization
for edge deployment. Runs latency benchmarks comparing:
  - PyTorch CPU inference
  - ONNX Runtime (FP32)  
  - ONNX Runtime (INT8 quantized)

Usage:
    python ml/export_edge.py

Output:
    models/rul_bilstm.onnx         — ONNX FP32 model
    models/rul_bilstm_int8.onnx    — ONNX INT8 quantized model
    models/benchmark_results.json  — Latency and size benchmarks
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────

PROJECT_ROOT = str(Path(__file__).parent.parent)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

N_BENCHMARK_RUNS = 200   # Number of inference runs for latency measurement
WARMUP_RUNS = 20         # Warmup runs before benchmarking


def load_pytorch_model():
    """Load the trained BiLSTM+Attention model."""
    from train_lstm import BiLSTMAttentionRUL
    
    checkpoint_path = os.path.join(MODEL_DIR, "rul_bilstm.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Model not found at {checkpoint_path}")
        print("  Run 'python ml/train_lstm.py' first to train the model.")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model = BiLSTMAttentionRUL(
        n_features=checkpoint['n_features'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def export_onnx(model, checkpoint):
    """Export PyTorch model to ONNX format."""
    print("\n[1/4] Exporting to ONNX...")
    
    n_features = checkpoint['n_features']
    seq_len = checkpoint['seq_len']
    
    # Create dummy input matching model's expected shape
    dummy_input = torch.randn(1, seq_len, n_features)
    
    onnx_path = os.path.join(MODEL_DIR, "rul_bilstm.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["sensor_sequence"],
        output_names=["rul_prediction"],
        dynamic_axes={
            "sensor_sequence": {0: "batch_size"},
            "rul_prediction": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter for cleaner graphs (quantization-compatible)
    )
    
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  [OK] ONNX FP32 exported: {onnx_path} ({size_mb:.2f} MB)")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  [OK] ONNX model validation passed")
    
    return onnx_path


def quantize_int8(onnx_path):
    """Apply dynamic INT8 quantization to ONNX model."""
    print("\n[2/4] Applying INT8 quantization...")
    
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    int8_path = os.path.join(MODEL_DIR, "rul_bilstm_int8.onnx")
    
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    
    fp32_size = os.path.getsize(onnx_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    compression = (1 - int8_size / fp32_size) * 100
    
    print(f"  [OK] INT8 model: {int8_path}")
    print(f"  FP32 size:  {fp32_size:.2f} MB")
    print(f"  INT8 size:  {int8_size:.2f} MB")
    print(f"  Compression: {compression:.1f}% smaller")
    
    return int8_path


def benchmark_pytorch(model, dummy_input):
    """Benchmark PyTorch CPU inference latency."""
    model.eval()
    latencies = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)
        
        # Benchmark
        for _ in range(N_BENCHMARK_RUNS):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return np.array(latencies)


def benchmark_onnx(onnx_path, dummy_input_np):
    """Benchmark ONNX Runtime inference latency."""
    import onnxruntime as ort
    
    # Use only CPU provider for fair comparison
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    
    input_name = session.get_inputs()[0].name
    latencies = []
    
    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = session.run(None, {input_name: dummy_input_np})
    
    # Benchmark
    for _ in range(N_BENCHMARK_RUNS):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input_np})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return np.array(latencies)


def verify_predictions(model, onnx_path, int8_path, checkpoint):
    """Verify ONNX and INT8 models produce similar outputs to PyTorch."""
    import onnxruntime as ort
    
    n_features = checkpoint['n_features']
    seq_len = checkpoint['seq_len']
    
    # Generate test input
    test_input = torch.randn(5, seq_len, n_features)
    test_input_np = test_input.numpy()
    
    # PyTorch prediction
    model.eval()
    with torch.no_grad():
        pytorch_out = model(test_input).numpy().flatten()
    
    # ONNX FP32 prediction
    sess_fp32 = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess_fp32.run(None, {"sensor_sequence": test_input_np})[0].flatten()
    
    # ONNX INT8 prediction
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    int8_out = sess_int8.run(None, {"sensor_sequence": test_input_np})[0].flatten()
    
    # Check closeness
    fp32_diff = np.max(np.abs(pytorch_out - onnx_out))
    int8_diff = np.max(np.abs(pytorch_out - int8_out))
    
    print(f"\n  Prediction Verification (5 samples):")
    print(f"  {'PyTorch':>12} | {'ONNX FP32':>12} | {'ONNX INT8':>12}")
    print(f"  {'-'*12} | {'-'*12} | {'-'*12}")
    for i in range(5):
        print(f"  {pytorch_out[i]:>12.2f} | {onnx_out[i]:>12.2f} | {int8_out[i]:>12.2f}")
    
    print(f"\n  Max |PyTorch - ONNX FP32|: {fp32_diff:.6f}")
    print(f"  Max |PyTorch - ONNX INT8|: {int8_diff:.6f}")
    
    if fp32_diff < 0.01:
        print("  [OK] ONNX FP32 matches PyTorch")
    else:
        print("  [WARN] ONNX FP32 diverges from PyTorch")
    
    if int8_diff < 2.0:
        print("  [OK] INT8 quantization error is acceptable")
    else:
        print("  [WARN] INT8 quantization error is high")


def run_benchmarks():
    """Full export + benchmark pipeline."""
    print("=" * 60)
    print("AeroGuard Edge Export & Benchmark Pipeline")
    print("=" * 60)
    
    # Load model
    model, checkpoint = load_pytorch_model()
    n_features = checkpoint['n_features']
    seq_len = checkpoint['seq_len']
    
    print(f"\n  Model: BiLSTM+Attention")
    print(f"  Features: {n_features}, Seq Length: {seq_len}")
    
    # Export
    onnx_path = export_onnx(model, checkpoint)
    int8_path = quantize_int8(onnx_path)
    
    # Verify
    print("\n[3/4] Verifying model predictions...")
    verify_predictions(model, onnx_path, int8_path, checkpoint)
    
    # Benchmark
    print("\n[4/4] Running latency benchmarks...")
    print(f"  Runs: {N_BENCHMARK_RUNS} (after {WARMUP_RUNS} warmup)")
    
    dummy_torch = torch.randn(1, seq_len, n_features)
    dummy_np = dummy_torch.numpy().astype(np.float32)
    
    # PyTorch CPU
    print("\n  Benchmarking PyTorch CPU...")
    pytorch_latencies = benchmark_pytorch(model, dummy_torch)
    
    # ONNX FP32
    print("  Benchmarking ONNX Runtime FP32...")
    onnx_latencies = benchmark_onnx(onnx_path, dummy_np)
    
    # ONNX INT8
    print("  Benchmarking ONNX Runtime INT8...")
    int8_latencies = benchmark_onnx(int8_path, dummy_np)
    
    # Results
    pth_size = os.path.getsize(os.path.join(MODEL_DIR, "rul_bilstm.pth")) / (1024 * 1024)
    fp32_size = os.path.getsize(onnx_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    
    results = {
        "pytorch_cpu": {
            "mean_ms": round(float(np.mean(pytorch_latencies)), 3),
            "median_ms": round(float(np.median(pytorch_latencies)), 3),
            "p95_ms": round(float(np.percentile(pytorch_latencies, 95)), 3),
            "p99_ms": round(float(np.percentile(pytorch_latencies, 99)), 3),
            "min_ms": round(float(np.min(pytorch_latencies)), 3),
            "max_ms": round(float(np.max(pytorch_latencies)), 3),
            "model_size_mb": round(pth_size, 3),
        },
        "onnx_fp32": {
            "mean_ms": round(float(np.mean(onnx_latencies)), 3),
            "median_ms": round(float(np.median(onnx_latencies)), 3),
            "p95_ms": round(float(np.percentile(onnx_latencies, 95)), 3),
            "p99_ms": round(float(np.percentile(onnx_latencies, 99)), 3),
            "min_ms": round(float(np.min(onnx_latencies)), 3),
            "max_ms": round(float(np.max(onnx_latencies)), 3),
            "model_size_mb": round(fp32_size, 3),
        },
        "onnx_int8": {
            "mean_ms": round(float(np.mean(int8_latencies)), 3),
            "median_ms": round(float(np.median(int8_latencies)), 3),
            "p95_ms": round(float(np.percentile(int8_latencies, 95)), 3),
            "p99_ms": round(float(np.percentile(int8_latencies, 99)), 3),
            "min_ms": round(float(np.min(int8_latencies)), 3),
            "max_ms": round(float(np.max(int8_latencies)), 3),
            "model_size_mb": round(int8_size, 3),
        },
        "speedup_onnx_vs_pytorch": round(float(np.mean(pytorch_latencies) / np.mean(onnx_latencies)), 2),
        "speedup_int8_vs_pytorch": round(float(np.mean(pytorch_latencies) / np.mean(int8_latencies)), 2),
        "compression_int8_vs_fp32_pct": round((1 - int8_size / fp32_size) * 100, 1),
        "n_benchmark_runs": N_BENCHMARK_RUNS,
    }
    
    # Save results
    results_path = os.path.join(MODEL_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 65)
    print("  AeroGuard Edge Inference Benchmarks")
    print("=" * 65)
    print(f"  {'Runtime':<22} | {'Mean':>8} | {'p95':>8} | {'p99':>8} | {'Size':>8}")
    print(f"  {'-'*22} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    
    for name, label in [("pytorch_cpu", "PyTorch CPU"), 
                         ("onnx_fp32", "ONNX FP32"),
                         ("onnx_int8", "ONNX INT8 (Edge)")]:
        r = results[name]
        print(f"  {label:<22} | {r['mean_ms']:>6.2f}ms | {r['p95_ms']:>6.2f}ms | {r['p99_ms']:>6.2f}ms | {r['model_size_mb']:>5.2f}MB")
    
    print(f"\n  Speedup (ONNX vs PyTorch):  {results['speedup_onnx_vs_pytorch']}x")
    print(f"  Speedup (INT8 vs PyTorch):  {results['speedup_int8_vs_pytorch']}x")
    print(f"  Size reduction (INT8/FP32): {results['compression_int8_vs_fp32_pct']}%")
    print("=" * 65)
    
    print(f"\n  Results saved: {results_path}")
    print("\n  [OK] Edge export pipeline complete!")
    
    return results


if __name__ == "__main__":
    run_benchmarks()
