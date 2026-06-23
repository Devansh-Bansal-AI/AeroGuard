# ✈️ AeroGuard — Edge-Native Aircraft Engine Health Intelligence

> *"Predicting turbofan failures at 35,000 feet — before they happen."*

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![React](https://img.shields.io/badge/React-19+-61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM+Attention-EE4C2C)
![ONNX](https://img.shields.io/badge/ONNX-INT8_Quantized-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Tests](https://img.shields.io/badge/Tests-12%20Passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Problem Statement

Aircraft engine failures cause **$150K+/hour** in AOG (Aircraft on Ground) costs. Current maintenance relies on scheduled intervals (MSG-3), missing early degradation signals. Cloud-based analytics fail because **there is no internet at 35,000 feet**.

**AeroGuard** deploys edge-native AI directly on aircraft systems to:
- 🧠 **Predict Remaining Useful Life (RUL)** of turbofan engines in real-time
- 🔍 **Detect anomalies** before they cascade into failures using dual ML models
- ✈️ **Operate offline** — zero cloud dependency during flight
- ⚡ **Run at the edge** — ONNX INT8 quantized inference at **0.35ms latency**
- 🔬 **Explain predictions** — attention-weight visualization shows *why* the model flagged an engine

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AeroGuard System                                │
│                                                                         │
│  ┌───────────────────┐   ┌──────────────────┐   ┌──────────────────┐   │
│  │  Edge AI Node      │   │  API Gateway     │   │  React Dashboard │   │
│  │  (Docker)          │──▶│  (Docker)        │──▶│  (Docker)        │   │
│  │                    │   │                  │   │                  │   │
│  │  • ONNX INT8 RUL   │   │  • FastAPI REST  │   │  • Fleet Monitor │   │
│  │  • IF + LSTM AE    │   │  • SSE Streaming │   │  • Diagnostics   │   │
│  │  • Feature Eng     │   │  • Explainability│   │  • Edge Metrics  │   │
│  │  • Redis Cache     │   │  • TimescaleDB   │   │  • Recharts Viz  │   │
│  └───────────────────┘   └──────────────────┘   └──────────────────┘   │
│                                                                         │
│  ML Pipeline (Offline Training)                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  C-MAPSS Data → 112 Features → BiLSTM+Attention → ONNX INT8   │    │
│  │  Synthetic Gen   Rolling Stats  Huber Loss         Quantized   │    │
│  │                  + Rate of Chg  + NASA Score        0.04 MB    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🧠 AI & ML Pipeline
| Component | Technology | Details |
|-----------|-----------|---------|
| **RUL Predictor** | BiLSTM + Temporal Attention | Predicts remaining engine cycles; RMSE 41.26 on synthetic data |
| **Anomaly Detector** | Isolation Forest (40%) + LSTM Autoencoder (60%) | Dual-approach: statistical + deep learning combined score |
| **Edge Inference** | ONNX INT8 Quantization | 4.27x speedup, 62.7% size reduction (0.04 MB), sub-millisecond |
| **Explainability** | Attention Weight Extraction | Shows which timesteps the model focused on for each prediction |

### 🖥️ Multi-Page Dashboard
| Page | Purpose |
|------|---------|
| **Fleet Dashboard** | Real-time fleet monitoring with radial gauges, sparklines, sensor drift alerts |
| **Diagnostic Cockpit** | RUL trend charts, attention heatmap, sensor radar, engine vitals |
| **Edge Metrics** | ONNX benchmarks, model size comparison, deployment pipeline visualization |

### 🔌 Backend API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/fleet` | GET | Fleet overview with all engine statuses |
| `/api/engine/{id}` | GET | Detailed engine status with sensor trends |
| `/api/engine/{id}/history` | GET | Sensor reading history from Redis cache |
| `/api/engine/{id}/db-history` | GET | Persisted history from TimescaleDB |
| `/api/engine/{id}/explain` | GET | Attention-based prediction explainability |
| `/api/benchmark` | GET | Edge inference benchmark results |
| `/api/model/info` | GET | Model metadata and configuration |
| `/api/telemetry` | POST | Manual telemetry ingestion |
| `/ws/stream` | SSE | Real-time fleet telemetry stream (500ms) |

---

## 📁 Project Structure

```
AeroGuard/
├── backend/                        # FastAPI backend
│   ├── main.py                     # App entry + REST + SSE streaming
│   ├── Dockerfile                  # Backend container
│   ├── db/                         # Database layer
│   │   ├── database.py             # SQLModel engine + init
│   │   └── models.py              # SensorReadingHistory schema
│   └── services/
│       ├── engine_simulator.py     # Real-time 4-engine fleet simulator
│       └── inference_service.py    # ONNX/PyTorch inference + anomaly + explainability
├── ml/                             # ML pipeline
│   ├── preprocess.py               # C-MAPSS preprocessing + synthetic data gen
│   ├── train_lstm.py               # BiLSTM + Attention RUL model (PyTorch)
│   ├── anomaly.py                  # Isolation Forest + LSTM Autoencoder
│   └── export_edge.py             # ONNX export + INT8 quantization + benchmark
├── frontend/                       # React 19 dashboard
│   ├── Dockerfile                  # Frontend container
│   └── src/
│       ├── App.jsx                 # Fleet Dashboard + routing
│       ├── main.jsx                # React Router entry point
│       ├── utils.js                # Shared constants and utilities
│       ├── index.css               # Cockpit-inspired dark theme (1300+ lines)
│       └── pages/
│           ├── DiagnosticCockpit.jsx  # Recharts: RUL trends, attention heatmap, radar
│           └── EdgeMetrics.jsx        # Recharts: latency bars, model size, pipeline
├── models/                         # Saved model artifacts
│   ├── rul_bilstm.pth             # PyTorch checkpoint (108 KB)
│   ├── rul_bilstm.onnx            # ONNX FP32 (105 KB)
│   ├── rul_bilstm_int8.onnx       # ONNX INT8 quantized (39 KB)
│   ├── isolation_forest.joblib     # Anomaly: Isolation Forest (2.3 MB)
│   ├── autoencoder.pth            # Anomaly: LSTM Autoencoder (515 KB)
│   ├── model_metadata.json        # Training metrics (RMSE, NASA Score)
│   ├── anomaly_metadata.json      # Anomaly model config
│   └── benchmark_results.json     # Edge inference benchmarks
├── tests/                          # Test suite (12 tests)
│   ├── test_api.py                # API endpoint tests (5 tests)
│   ├── test_model.py              # Model quality + ONNX integrity (6 tests)
│   └── test_preprocess.py         # Data pipeline test (1 test)
├── docker-compose.yml              # 4-service orchestration
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Option A: Docker Compose (Recommended)
```bash
docker-compose up --build
# Dashboard: http://localhost:3000
# API:       http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

### Option B: Local Development

#### Prerequisites
- Python 3.10+
- Node.js 18+
- Redis (optional — simulator works without it)

#### 1. Install & Train
```bash
pip install -r requirements.txt

# Train the complete ML pipeline
python ml/preprocess.py         # Generate synthetic C-MAPSS data + feature engineering
python ml/train_lstm.py         # Train BiLSTM+Attention RUL model
python ml/anomaly.py            # Train Isolation Forest + LSTM Autoencoder
python ml/export_edge.py        # Export ONNX + INT8 quantization + benchmarks
```

#### 2. Start Backend
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

#### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
```

---

## 📊 Edge AI Performance

| Metric | PyTorch CPU | ONNX FP32 | ONNX INT8 |
|--------|------------|-----------|-----------|
| **Mean Latency** | 2.35 ms | 0.74 ms | **0.55 ms** |
| **P95 Latency** | 4.12 ms | 1.21 ms | **0.89 ms** |
| **Model Size** | 108 KB | 105 KB | **39 KB** |
| **Speedup** | 1.0x | 3.2x | **4.27x** |

> INT8 quantization achieves **62.7% size reduction** with **<0.1 cycle** prediction drift vs PyTorch — verified by automated tests.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Expected output: 12 passed
# - test_api.py: 5 tests (health, fleet, engine, 404, benchmark)
# - test_model.py: 6 tests (metadata, RMSE, ONNX exist, compression, INT8 accuracy, benchmarks)
# - test_preprocess.py: 1 test (full pipeline)
```

---

## 📐 Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)
- Turbofan engine run-to-failure degradation data
- 21 sensor channels + 3 operational settings
- **Built-in synthetic generator** — no external download required
- Supports real C-MAPSS FD001–FD004 datasets for research validation

---

## 🔬 Explainability

AeroGuard implements **temporal attention explainability** — a feature rarely seen in predictive maintenance systems:

```
GET /api/engine/1/explain

Response:
{
  "rul_prediction": 87.5,
  "attention_weights": [0.018, 0.019, ..., 0.034],
  "top_timesteps": [
    {"timestep": 48, "weight": 0.0342},
    {"timestep": 47, "weight": 0.0301}
  ],
  "interpretation": "Higher attention weight = model focused more on that timestep.
                      Recent timesteps with high weights indicate acute degradation."
}
```

The Diagnostic Cockpit visualizes these weights as an interactive heatmap, proving *why* the model flagged an engine — critical for aerospace regulatory compliance.

---

## 🐳 Infrastructure

```yaml
# docker-compose.yml — 4 services
services:
  edge-api:      # FastAPI + ONNX Runtime (256MB limit, 0.5 CPU)
  dashboard:     # React 19 + Vite
  redis:         # Real-time sensor cache
  timescaledb:   # Persistent telemetry storage (PostgreSQL)
```

Resource constraints (`256MB RAM, 0.5 CPU`) simulate real edge hardware limitations.

---

## 👥 Team

| Role | Responsibility |
|------|---------------|
| ML Lead | BiLSTM training, evaluation, NASA scoring, anomaly detection |
| Edge AI Engineer | ONNX export, INT8 quantization, latency benchmarking |
| Backend Engineer | FastAPI, SSE streaming, explainability API, TimescaleDB |
| Frontend Engineer | React dashboard, Recharts visualizations, multi-page routing |
| DevOps + Research | Docker, data pipeline, testing, documentation |

---

## 📄 License

MIT License
