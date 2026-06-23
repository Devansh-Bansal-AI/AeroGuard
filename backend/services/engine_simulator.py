"""
AeroGuard — Engine Simulator

Simulates real-time turbofan engine telemetry for 4 engines in a fleet.
Each engine progresses through its lifecycle with realistic degradation.

The simulator produces sensor readings that mirror the NASA C-MAPSS format,
allowing the ML models to make real-time predictions during demos.

Engine lifecycle phases:
1. Healthy: Normal operation, low noise, stable readings
2. Early Degradation: Subtle drift in key sensors
3. Advanced Degradation: Clear trends, increasing noise
4. Critical: Imminent failure, anomalous readings, spikes
"""

import numpy as np
import time
import json
import redis
import os
from typing import Optional
from datetime import datetime


class EngineState:
    """State container for a single engine."""
    
    def __init__(self, engine_id: int, max_life: int, seed: int = None):
        self.engine_id = engine_id
        self.max_life = max_life
        self.current_cycle = 0
        self.rng = np.random.RandomState(seed)
        
        # State is now in Redis, but we keep metadata here
        # self.history = deque(maxlen=200)
        
        # Latest prediction results
        self.rul_prediction = float(max_life)
        self.anomaly_score = 0.0
        self.is_anomaly = False
        self.health_status = "healthy"
        self.trend = "stable"
        
        # Sensor baselines (realistic C-MAPSS values)
        self.sensor_baselines = {
            "T2":    518.67,   # Total temperature at fan inlet
            "T24":   642.44,   # Total temperature at LPC outlet
            "T30":   1589.7,   # Total temperature at HPC outlet
            "T50":   1400.6,   # Total temperature at LPT outlet
            "P2":    14.62,    # Pressure at fan inlet
            "P15":   21.61,    # Total pressure in bypass-duct
            "P30":   554.36,   # Total pressure at HPC outlet
            "Nf":    2388.0,   # Physical fan speed
            "Nc":    9046.2,   # Physical core speed
            "Ps30":  47.47,    # Static pressure at HPC outlet
            "phi":   521.66,   # Ratio of fuel flow to Ps30
            "NRf":   2388.0,   # Corrected fan speed
            "NRc":   8138.6,   # Corrected core speed
            "BPR":   8.449,    # Bypass ratio
        }
        
        # Drift rates per sensor (how much they change over life)
        self.drift_rates = {
            "T2":    0.0,    "T24":   0.8,    "T30":   2.5,
            "T50":   6.0,    "P2":    0.0,    "P15":   0.0,
            "P30":   1.5,    "Nf":    0.5,    "Nc":    3.0,
            "Ps30":  0.15,   "phi":   0.7,    "NRf":   0.5,
            "NRc":   2.5,    "BPR":   0.02,
        }
        
        # Noise levels per sensor
        self.noise_levels = {
            "T2":    0.0,    "T24":   0.5,    "T30":   1.2,
            "T50":   3.0,    "P2":    0.0,    "P15":   0.0,
            "P30":   0.8,    "Nf":    0.2,    "Nc":    2.0,
            "Ps30":  0.05,   "phi":   0.4,    "NRf":   0.2,
            "NRc":   1.5,    "BPR":   0.005,
        }


class EngineSimulator:
    """
    Fleet engine simulator for real-time dashboard demos.
    
    Simulates 4 engines at different points in their lifecycle:
    - Engine 1: Healthy (early lifecycle)
    - Engine 2: Early degradation (mid lifecycle)
    - Engine 3: Advanced degradation (late lifecycle)
    - Engine 4: Critical (near failure)
    
    Args:
        n_engines: Number of engines to simulate
        inference_service: Optional ML inference service for predictions
    """
    
    def __init__(self, n_engines: int = 4, inference_service=None, db_engine=None):
        self.inference_service = inference_service
        self.db_engine = db_engine  # SQLAlchemy/SQLModel engine for persistence
        
        # Create engines at different lifecycle stages for interesting demo
        engine_configs = [
            {"engine_id": 1, "max_life": 320, "start_cycle": 30,  "name": "CFM56-7B #1"},
            {"engine_id": 2, "max_life": 280, "start_cycle": 140, "name": "CFM56-7B #2"},
            {"engine_id": 3, "max_life": 250, "start_cycle": 200, "name": "CFM56-7B #3"},
            {"engine_id": 4, "max_life": 220, "start_cycle": 195, "name": "CFM56-7B #4"},
        ]
        
        # Connect to Redis
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        try:
            self.redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
        except redis.ConnectionError:
            try:
                # Fallback to docker host name just in case
                self.redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
                self.redis_client.ping()
            except redis.ConnectionError:
                print(f"[WARN] Redis not available at {redis_host} or 'redis', falling back to local memory simulation (hacky).")
                self.redis_client = None
            
        self.engines = {}
        for i, config in enumerate(engine_configs[:n_engines]):
            engine = EngineState(
                engine_id=config["engine_id"],
                max_life=config["max_life"],
                seed=42 + i
            )
            engine.current_cycle = config["start_cycle"]
            engine.name = config["name"]
            self.engines[config["engine_id"]] = engine
            
            # Initialize Redis state for this engine
            if self.redis_client:
                # Clear old data
                self.redis_client.delete(f"engine:{engine.engine_id}:history")
            
            # Pre-fill some history
            for _ in range(50):
                reading = self._generate_reading(engine)
                self._save_reading(engine.engine_id, reading)
                engine.current_cycle += 1
                
    def _save_reading(self, engine_id: int, reading: dict):
        # Save to Redis (real-time cache)
        if self.redis_client:
            self.redis_client.rpush(f"engine:{engine_id}:history", json.dumps(reading))
            # Keep only last 200 readings
            self.redis_client.ltrim(f"engine:{engine_id}:history", -200, -1)
        
        # Persist to database (best-effort, non-blocking)
        if self.db_engine is not None:
            try:
                from sqlmodel import Session
                from backend.db.models import SensorReadingHistory
                
                record = SensorReadingHistory(
                    engine_id=reading["engine_id"],
                    cycle=reading["cycle"],
                    timestamp=datetime.utcnow(),
                    rul_prediction=reading["rul_prediction"],
                    anomaly_score=reading["anomaly_score"],
                    is_anomaly=reading["is_anomaly"],
                    health_status=reading["health_status"],
                    sensors_json=json.dumps(reading["sensors"]),
                )
                with Session(self.db_engine) as session:
                    session.add(record)
                    session.commit()
            except Exception:
                pass  # Don't let DB errors break the simulator
            
    def _get_history(self, engine_id: int) -> list:
        if self.redis_client:
            data = self.redis_client.lrange(f"engine:{engine_id}:history", 0, -1)
            return [json.loads(d) for d in data]
        return []
        
    def inject_reading(self, engine_id: int, manual_sensors: dict):
        engine = self.engines.get(engine_id)
        if not engine:
            return
        
        engine.current_cycle += 1
        reading = self._generate_reading(engine)
        # Override with manual sensors (allows injecting NaNs)
        reading["sensors"].update(manual_sensors)
        self._save_reading(engine_id, reading)
    
    def _generate_reading(self, engine: EngineState) -> dict:
        """Generate a single sensor reading for an engine."""
        progress = engine.current_cycle / engine.max_life
        degradation = progress ** 2.3
        
        # Add accelerated degradation in final 20%
        if progress > 0.8:
            degradation += (progress - 0.8) * 2.5
        
        sensors = {}
        for name, baseline in engine.sensor_baselines.items():
            drift = engine.drift_rates[name]
            noise = engine.noise_levels[name]
            
            value = baseline + drift * degradation * engine.max_life * 0.01
            value += engine.rng.normal(0, noise)
            
            # Anomalous spikes near end of life
            if progress > 0.85 and engine.rng.random() < 0.08:
                value += drift * 4 * engine.rng.random()
            
            sensors[name] = round(float(value), 4)
        
        # Calculate health metrics
        if progress < 0.5:
            health_status = "healthy"
            trend = "stable"
        elif progress < 0.75:
            health_status = "healthy"
            trend = "stable"
        elif progress < 0.85:
            health_status = "warning"
            trend = "degrading"
        else:
            health_status = "critical"
            trend = "critical_degradation"
        
        # RUL estimation — USE REAL MODEL IF AVAILABLE
        true_rul = max(0, engine.max_life - engine.current_cycle)
        
        history = self._get_history(engine.engine_id)
        prediction = None  # Will hold model output (RUL + anomaly) if available
        
        if (self.inference_service and 
            self.inference_service.is_ready and 
            self.inference_service.mode != "simulated" and
            len(history) >= 49):
            # Build sensor history for model inference
            sensor_hist = [r["sensors"] for r in history]
            sensor_hist.append(sensors)  # Include current reading
            prediction = self.inference_service.predict_from_sensors(
                sensor_hist, fallback_rul=true_rul
            )
            predicted_rul = prediction["rul"]
            model_used = prediction["model_used"]
        else:
            # Fallback: ground truth with noise (when model not trained)
            predicted_rul = max(0, true_rul + engine.rng.normal(0, 5))
            model_used = "simulated"
        
        # Anomaly detection — use real model output if available
        if (prediction is not None and 
            prediction.get("anomaly_score") is not None):
            # Real anomaly model output (IF + LSTM Autoencoder)
            anomaly_score = float(prediction["anomaly_score"])
            is_anomaly = bool(prediction.get("is_anomaly", anomaly_score > 0.65))
        else:
            # Heuristic fallback (increases with degradation)
            base_anomaly = min(1.0, degradation * 0.8)
            anomaly_noise = engine.rng.normal(0, 0.05)
            anomaly_score = max(0, min(1.0, base_anomaly + anomaly_noise))
            is_anomaly = anomaly_score > 0.65 or predicted_rul < 30
        
        return {
            "engine_id": engine.engine_id,
            "cycle": engine.current_cycle,
            "sensors": sensors,
            "rul_prediction": round(float(predicted_rul), 1),
            "anomaly_score": round(float(anomaly_score), 4),
            "is_anomaly": is_anomaly,
            "health_status": health_status,
            "trend": trend,
            "model_used": model_used,
            "timestamp": time.time(),
        }
    
    def step(self):
        """Advance all engines by one cycle."""
        for engine_id, engine in self.engines.items():
            engine.current_cycle += 1
            
            # Reset engine if it reaches end of life (continuous demo)
            if engine.current_cycle >= engine.max_life:
                engine.current_cycle = 30  # Reset to early life
                if self.redis_client:
                    self.redis_client.delete(f"engine:{engine_id}:history")
            
            reading = self._generate_reading(engine)
            self._save_reading(engine_id, reading)
            
            # Update engine state
            engine.rul_prediction = reading["rul_prediction"]
            engine.anomaly_score = reading["anomaly_score"]
            engine.is_anomaly = reading["is_anomaly"]
            engine.health_status = reading["health_status"]
            engine.trend = reading["trend"]
    
    def get_fleet_status(self) -> list:
        """Get current status of all engines."""
        statuses = []
        for engine_id, engine in self.engines.items():
            history = self._get_history(engine_id)
            if history:
                latest = history[-1]
                
                # Identify top drifting sensors for root-cause analysis
                top_drifting = []
                if len(history) >= 10:
                    recent = history[-10:]
                    drifts = {}
                    for sensor_name in engine.sensor_baselines.keys():
                        vals = [r["sensors"][sensor_name] for r in recent]
                        if len(vals) >= 2:
                            slope = np.polyfit(range(len(vals)), vals, 1)[0]
                            baseline = engine.sensor_baselines[sensor_name]
                            # Normalize slope by baseline magnitude
                            norm_drift = abs(slope) / (abs(baseline) + 1e-6)
                            drifts[sensor_name] = norm_drift
                    # Top 3 drifting sensors
                    top_drifting = sorted(drifts.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_drifting = [{"sensor": s, "drift_rate": round(d * 1000, 4)} for s, d in top_drifting]
                
                statuses.append({
                    "engine_id": engine.engine_id,
                    "name": getattr(engine, "name", f"Engine #{engine.engine_id}"),
                    "current_cycle": engine.current_cycle,
                    "max_life": engine.max_life,
                    "life_progress": round(engine.current_cycle / engine.max_life * 100, 1),
                    "rul_prediction": engine.rul_prediction,
                    "anomaly_score": engine.anomaly_score,
                    "is_anomaly": engine.is_anomaly,
                    "health_status": engine.health_status,
                    "trend": engine.trend,
                    "model_used": latest.get("model_used", "simulated"),
                    "sensors": latest.get("sensors", {}),
                    "top_drifting_sensors": top_drifting,
                })
        return statuses
    
    def get_engine_status(self, engine_id: int) -> Optional[dict]:
        """Get detailed status for a specific engine."""
        engine = self.engines.get(engine_id)
        if not engine:
            return None
        
        history = self._get_history(engine_id)
        if not history:
            # Return basic state even without history (e.g. Redis is down)
            return {
                "engine_id": engine.engine_id,
                "name": getattr(engine, "name", f"Engine #{engine.engine_id}"),
                "current_cycle": engine.current_cycle,
                "max_life": engine.max_life,
                "life_progress": round(engine.current_cycle / engine.max_life * 100, 1),
                "rul_prediction": engine.rul_prediction,
                "anomaly_score": engine.anomaly_score,
                "is_anomaly": engine.is_anomaly,
                "health_status": engine.health_status,
                "trend": engine.trend,
                "model_used": "simulated",
                "sensors": {},
                "sensor_trends": {},
                "history_length": 0,
            }
        
        latest = history[-1]
        
        # Calculate sensor trends (last 10 readings)
        sensor_trends = {}
        if len(history) >= 10:
            recent = history[-10:]
            for sensor_name in engine.sensor_baselines.keys():
                values = [r["sensors"][sensor_name] for r in recent]
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                sensor_trends[sensor_name] = {
                    "current": values[-1],
                    "trend_slope": round(float(trend_slope), 6),
                    "min_10": round(float(min(values)), 4),
                    "max_10": round(float(max(values)), 4),
                }
        
        return {
            "engine_id": engine.engine_id,
            "name": getattr(engine, "name", f"Engine #{engine.engine_id}"),
            "current_cycle": engine.current_cycle,
            "max_life": engine.max_life,
            "life_progress": round(engine.current_cycle / engine.max_life * 100, 1),
            "rul_prediction": engine.rul_prediction,
            "anomaly_score": engine.anomaly_score,
            "is_anomaly": engine.is_anomaly,
            "health_status": engine.health_status,
            "trend": engine.trend,
            "sensors": latest.get("sensors", {}),
            "sensor_trends": sensor_trends,
        }
    
    def get_engine_history(self, engine_id: int, last_n: int = 100) -> Optional[dict]:
        """Get historical readings for an engine."""
        engine = self.engines.get(engine_id)
        if not engine:
            return None
        
        history = self._get_history(engine_id)[-last_n:]
        
        return {
            "engine_id": engine.engine_id,
            "name": getattr(engine, "name", f"Engine #{engine.engine_id}"),
            "readings": history,
            "count": len(history),
        }
