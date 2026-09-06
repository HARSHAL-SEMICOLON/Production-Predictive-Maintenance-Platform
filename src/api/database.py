"""Database configuration and ORM models for Sentinel AI (PostgreSQL / SQLite)."""

from datetime import datetime
from typing import Generator
import json

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.config import settings
from src.utils.logger import logger

Base = declarative_base()


class SensorReading(Base):
    """Stores raw sensor telemetry received from factory IoT edge nodes."""
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    engine_id = Column(Integer, index=True, nullable=False)
    cycle = Column(Integer, index=True, nullable=False)
    
    # Settings
    setting_1 = Column(Float, default=0.0)
    setting_2 = Column(Float, default=0.0)
    setting_3 = Column(Float, default=100.0)

    # Sensors 1 to 21
    sensor_1 = Column(Float)
    sensor_2 = Column(Float)
    sensor_3 = Column(Float)
    sensor_4 = Column(Float)
    sensor_5 = Column(Float)
    sensor_6 = Column(Float)
    sensor_7 = Column(Float)
    sensor_8 = Column(Float)
    sensor_9 = Column(Float)
    sensor_10 = Column(Float)
    sensor_11 = Column(Float)
    sensor_12 = Column(Float)
    sensor_13 = Column(Float)
    sensor_14 = Column(Float)
    sensor_15 = Column(Float)
    sensor_16 = Column(Float)
    sensor_17 = Column(Float)
    sensor_18 = Column(Float)
    sensor_19 = Column(Float)
    sensor_20 = Column(Float)
    sensor_21 = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Stores inference outputs and operational predictions."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    engine_id = Column(Integer, index=True, nullable=False)
    cycle = Column(Integer, index=True, nullable=False)
    predicted_rul = Column(Float, nullable=False)
    failure_probability = Column(Float, nullable=False)
    health_status = Column(String(20), nullable=False)  # HEALTHY, WARNING, CRITICAL
    model_used = Column(String(50), nullable=False)
    latency_ms = Column(Float)
    top_drivers = Column(Text)  # JSON string of top feature attributions
    created_at = Column(DateTime, default=datetime.utcnow)


class AlertLog(Base):
    """Stores critical predictive maintenance warnings and notifications."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    engine_id = Column(Integer, index=True, nullable=False)
    cycle = Column(Integer, nullable=False)
    alert_level = Column(String(20), nullable=False)  # WARNING, CRITICAL
    message = Column(String(255), nullable=False)
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    """Stores model training experiments and champion promotion history."""
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)
    rmse = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    r2_score = Column(Float, nullable=False)
    nasa_score = Column(Float)
    is_champion = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Initialize Database Engine with seamless SQLite fallback
def get_engine():
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
        )
        # Test connection
        with engine.connect() as conn:
            pass
        return engine
    except Exception as e:
        logger.warning(
            f"Could not connect to {settings.DATABASE_URL} ({e}). Falling back to SQLite local database."
        )
        sqlite_url = "sqlite:///./sentinel_maintenance.db"
        return create_engine(sqlite_url, connect_args={"check_same_thread": False})


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Creates database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Initialized database schema.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
