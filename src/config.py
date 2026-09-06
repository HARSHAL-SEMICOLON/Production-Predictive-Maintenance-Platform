"""Production configuration module for Sentinel AI Predictive Maintenance Platform."""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "Sentinel AI - Predictive Maintenance Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    DATA_EXTERNAL_DIR: Path = BASE_DIR / "data" / "external"
    MODELS_DIR: Path = BASE_DIR / "models" / "artifacts"
    REPORTS_DIR: Path = BASE_DIR / "reports"

    # Database & Cache Configuration
    # Defaults to SQLite for immediate local execution without external DB dependencies
    DATABASE_URL: str = "sqlite:///./sentinel_maintenance.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = "sqlite:///./mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "sentinel_predictive_maintenance"

    # C-MAPSS Dataset Feature Schema
    SETTINGS_COLS: List[str] = ["setting_1", "setting_2", "setting_3"]
    ALL_SENSOR_COLS: List[str] = [f"sensor_{i}" for i in range(1, 22)]
    
    # Informative sensors (standard NASA C-MAPSS FD001 non-constant channels)
    # Channels 1, 5, 6, 10, 16, 18, 19 have virtually zero variance in FD001
    INFORMATIVE_SENSOR_COLS: List[str] = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
        "sensor_15", "sensor_17", "sensor_20", "sensor_21"
    ]

    # Sensor Domain Descriptions for Industrial SCADA UI
    SENSOR_DESCRIPTIONS: dict = {
        "sensor_1": "Fan Inlet Temp (T2) [°R]",
        "sensor_2": "LPC Outlet Temp (T24) [°R]",
        "sensor_3": "HPC Outlet Temp (T30) [°R]",
        "sensor_4": "LPT Outlet Temp (T50) [°R]",
        "sensor_5": "Fan Inlet Pressure (P2) [psia]",
        "sensor_6": "Bypass Duct Pressure (P15) [psia]",
        "sensor_7": "HPC Outlet Pressure (P30) [psia]",
        "sensor_8": "Physical Fan Speed (Nf) [rpm]",
        "sensor_9": "Physical Core Speed (Nc) [rpm]",
        "sensor_10": "Engine Pressure Ratio (epr)",
        "sensor_11": "HPC Static Pressure (Ps30) [psia]",
        "sensor_12": "Burner Fuel-Air Ratio (phi)",
        "sensor_13": "Corrected Fan Speed (NRf) [rpm]",
        "sensor_14": "Corrected Core Speed (NRc) [rpm]",
        "sensor_15": "Bypass Ratio (BPR)",
        "sensor_16": "Burner Fuel Flow (fuel_flow) [pps]",
        "sensor_17": "Bleed Enthalpy (htBleed)",
        "sensor_18": "Demanded Fan Speed (Nf_dmd) [rpm]",
        "sensor_19": "Demanded Corrected Fan Speed (PCNfR_dmd)",
        "sensor_20": "HPT Coolant Bleed (W31) [lbm/s]",
        "sensor_21": "LPT Coolant Bleed (W32) [lbm/s]",
    }

    # Time-Series Windowing Parameters
    SEQUENCE_WINDOW_SIZE: int = 30
    RUL_CLIP_THRESHOLD: int = 125  # Piecewise linear clipping standard for C-MAPSS
    
    # Maintenance Risk Thresholds (Cycles Remaining)
    CRITICAL_RUL_THRESHOLD: int = 30   # Immediate maintenance required (<30 cycles)
    WARNING_RUL_THRESHOLD: int = 75    # Schedule maintenance (<75 cycles)
    HEALTHY_RUL_THRESHOLD: int = 120   # Normal operation

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure directories exist
settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
