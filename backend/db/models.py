from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class SensorReadingHistory(SQLModel, table=True):
    """Historical record of engine telemetry and model predictions."""
    id: Optional[int] = Field(default=None, primary_key=True)
    engine_id: int = Field(index=True)
    cycle: int
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Model Outputs
    rul_prediction: float
    anomaly_score: float
    is_anomaly: bool
    health_status: str
    
    # Raw Sensors (Stored as JSON string for flexibility)
    sensors_json: str
