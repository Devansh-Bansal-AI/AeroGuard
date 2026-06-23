import os
from sqlmodel import create_engine, SQLModel

# In production this should come from environment variables.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aeroguard:aeroguard_pass@localhost:5432/aeroguard_tsdb")

# For local development without docker, fallback to sqlite if postgres is unavailable.
# But since we require timescaledb/postgres, we should prefer connecting to it.
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
except Exception as e:
    # Fallback to sqlite if needed, but ideally we want postgres.
    engine = create_engine("sqlite:///aeroguard.db")

def init_db():
    """Create database tables."""
    from backend.db.models import SensorReadingHistory
    SQLModel.metadata.create_all(engine)

def get_session():
    """Dependency to provide a database session."""
    from sqlmodel import Session
    with Session(engine) as session:
        yield session
