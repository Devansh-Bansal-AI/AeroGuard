import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from backend.main import app

def test_health_check():
    """Test the health check endpoint."""
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data

def test_get_fleet_status():
    """Test the fleet overview endpoint."""
    with TestClient(app) as client:
        response = client.get("/api/fleet")
        assert response.status_code == 200
        data = response.json()
        assert "total_engines" in data
        assert "healthy_count" in data
        assert "warning_count" in data
        assert "critical_count" in data
        assert "engines" in data
        assert isinstance(data["engines"], list)

def test_get_engine_status_valid():
    """Test getting status for a valid engine."""
    with TestClient(app) as client:
        # Assuming engine 1 is always created by the simulator
        response = client.get("/api/engine/1")
        assert response.status_code == 200
        data = response.json()
        assert data["engine_id"] == 1
        assert "rul_prediction" in data

def test_get_engine_status_invalid():
    """Test getting status for an invalid engine."""
    with TestClient(app) as client:
        response = client.get("/api/engine/999")
        assert response.status_code == 404

def test_get_benchmark_results():
    """Test the benchmark results endpoint."""
    with TestClient(app) as client:
        response = client.get("/api/benchmark")
        assert response.status_code == 200
        data = response.json()
        assert "pytorch_cpu" in data
        assert "onnx_fp32" in data
        assert "onnx_int8" in data
