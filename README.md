# ✈️ AeroGuard — Edge-Native Aircraft Engine Health Intelligence

> *"Predicting turbofan failures at 35,000 feet — before they happen."*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![React](https://img.shields.io/badge/React-19+-61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM+Attention-EE4C2C)
![ONNX](https://img.shields.io/badge/ONNX-INT8_Edge_AI-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Problem Statement

Aircraft engine failures cause **$150K+/hour** in AOG (Aircraft on Ground) costs. Current maintenance relies on scheduled intervals (MSG-3), missing early degradation signals. Cloud-based analytics fail because **there is no internet at 35,000 feet**.

**AeroGuard** deploys edge-native AI directly on aircraft systems to:
- **Predict Remaining Useful Life (RUL)** of turbofan engines in real-time
- **Detect anomalies** before they cascade into failures
- **Operate offline** — zero cloud dependency during flight
- **Run at the edge** — ONNX INT8 quantized inference

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       AeroGuard System                           │
│                                                                  │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │  Edge Node       │   │  API Gateway     │   │  Dashboard   │ │
│  │  (Docker)        │──▶│  (Docker)        │──▶│  (Docker)    │ │
│  │                  │   │                  │   │              │ │
│  │  • Sensor Sim    │   │  • FastAPI       │   │  • React     │ │
│  │  • ONNX Runtime  │   │  • SSE Streaming │   │  • Gauges    │ │
│  │  • INT8 Quantize │   │  • REST API      │   │  • Sparkline │ │
│  │  • Redis State   │   │  • Fleet Mgmt    │   │  • Alerts    │ │
│  └─────────────────┘   └──────────────────┘   └──────────────┘ │
│                                                                  │
│  ML Pipeline (Offline)                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  C-MAPSS Data → Feature Eng → BiLSTM+Attention → ONNX INT8 │ │
│  │  Synthetic Gen   Rolling Stats  Huber Loss       Quantized  │ │
│  │                  + Rate of Chg  + NASA Score                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
AeroGuard/
├── backend/                    # FastAPI backend
│   ├── main.py                 # App entry + SSE streaming
│   ├── Dockerfile              # Backend container
│   └── services/
│       ├── engine_simulator.py # Real-time 4-engine fleet simulator
│       └── inference_service.py# ONNX/PyTorch/XGBoost model inference
├── ml/                         # ML pipeline
│   ├── preprocess.py           # C-MAPSS preprocessing + synthetic data gen
│   ├── train_lstm.py           # BiLSTM + Attention RUL model (PyTorch)
│   ├── train_xgboost.py        # XGBoost RUL model (legacy edge baseline)
│   ├── anomaly.py              # Isolation Forest + LSTM Autoencoder
│   └── export_edge.py          # ONNX export + INT8 quantization + benchmark
├── frontend/                   # React cockpit dashboard
│   ├── Dockerfile              # Frontend container
│   └── src/
│       ├── App.jsx             # Dashboard with gauges, sparklines, alerts
│       └── index.css           # Cockpit-inspired dark theme
├── docker-compose.yml          # One-command deployment
├── data/                       # Raw & processed data
├── models/                     # Saved models (.pth, .onnx)
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Option A: Docker (Recommended)
```bash
docker-compose up --build
# Dashboard: http://localhost:3000
# API: http://localhost:8000
```

### Option B: Local Development

#### Prerequisites
- Python 3.10+
- Node.js 18+

#### 1. Install & Train
```bash
pip install -r requirements.txt
python ml/preprocess.py         # Generate synthetic C-MAPSS data
python ml/train_lstm.py         # Train BiLSTM+Attention RUL model
python ml/export_edge.py        # Export to ONNX + INT8 quantize + benchmark
```

#### 2. Start Backend
```bash
python backend/main.py
# API running on http://localhost:8000
```

#### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
# Dashboard running on http://localhost:5173
```

## 📊 Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)
- Turbofan engine run-to-failure degradation data
- 21 sensor channels + 3 operational settings
- **Built-in synthetic generator** — no external download required for demos
- Supports real C-MAPSS FD001–FD004 datasets for research

## 🧠 AI Models

| Model | Purpose | Architecture | Edge Ready |
|-------|---------|-------------|------------|
| RUL Predictor | Remaining Useful Life estimation | BiLSTM + Temporal Attention | ✅ ONNX INT8 |
| Anomaly Detector | Early fault detection | Isolation Forest + LSTM Autoencoder | ✅ |
| RUL Baseline | Edge-optimized baseline | XGBoost (legacy) | ✅ |

## 🏆 Hackathon Theme

**AI at the Edge Solutions for Aerospace** — Edge AI for Predictive Maintenance & Aircraft Health Monitoring

## 👥 Team Roles

| Role | Responsibility |
|------|---------------|
| ML Lead | BiLSTM training, evaluation, NASA scoring |
| Edge AI Engineer | ONNX export, INT8 quantization, latency benchmarking |
| Backend Engineer | FastAPI, SSE streaming, inference integration |
| Frontend Engineer | React cockpit dashboard, radial gauges, sparklines |
| DevOps + Research | Docker, data pipeline, demo scripting |

## 📄 License

MIT License
