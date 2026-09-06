"""Pydantic data validation schemas for Sentinel AI API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class SensorTelemetryIn(BaseModel):
    """Schema for incoming real-time IoT sensor telemetry."""
    engine_id: int = Field(..., ge=1, description="Unique Turbofan Engine ID")
    cycle: int = Field(..., ge=1, description="Operational Cycle sequence number")
    
    # Operational settings
    setting_1: float = Field(default=0.0, description="Altitude / Throttle Resolver Angle")
    setting_2: float = Field(default=0.0, description="Mach Number")
    setting_3: float = Field(default=100.0, description="Sea Level Static Temperature")

    # 21 C-MAPSS Sensor channels
    sensor_1: float = Field(default=518.67, ge=0.0, description="Fan Inlet Temp (T2)")
    sensor_2: float = Field(default=642.50, ge=400.0, le=800.0, description="LPC Outlet Temp (T24)")
    sensor_3: float = Field(default=1588.0, ge=1200.0, le=1800.0, description="HPC Outlet Temp (T30)")
    sensor_4: float = Field(default=1404.0, ge=1000.0, le=1700.0, description="LPT Outlet Temp (T50)")
    sensor_5: float = Field(default=14.62, ge=0.0, description="Fan Inlet Pressure (P2)")
    sensor_6: float = Field(default=21.61, ge=0.0, description="Bypass Duct Pressure (P15)")
    sensor_7: float = Field(default=553.8, ge=300.0, le=700.0, description="HPC Outlet Pressure (P30)")
    sensor_8: float = Field(default=2388.08, ge=2000.0, le=2600.0, description="Physical Fan Speed (Nf)")
    sensor_9: float = Field(default=9055.0, ge=8000.0, le=10000.0, description="Physical Core Speed (Nc)")
    sensor_10: float = Field(default=1.30, ge=0.5, le=2.5, description="Engine Pressure Ratio (epr)")
    sensor_11: float = Field(default=47.45, ge=30.0, le=65.0, description="HPC Static Pressure (Ps30)")
    sensor_12: float = Field(default=521.8, ge=400.0, le=600.0, description="Ratio of Fuel Flow to Ps30")
    sensor_13: float = Field(default=2388.09, ge=2000.0, le=2600.0, description="Corrected Fan Speed (NRf)")
    sensor_14: float = Field(default=8135.0, ge=7500.0, le=9000.0, description="Corrected Core Speed (NRc)")
    sensor_15: float = Field(default=8.42, ge=6.0, le=12.0, description="Bypass Ratio (BPR)")
    sensor_16: float = Field(default=0.03, ge=0.0, le=1.0, description="Burner Fuel Flow")
    sensor_17: float = Field(default=392.5, ge=300.0, le=500.0, description="Bleed Enthalpy (htBleed)")
    sensor_18: float = Field(default=2388.0, ge=2000.0, le=2600.0, description="Demanded Fan Speed")
    sensor_19: float = Field(default=100.0, ge=80.0, le=120.0, description="Demanded Corrected Fan Speed")
    sensor_20: float = Field(default=38.85, ge=25.0, le=50.0, description="HPT Coolant Bleed (W31)")
    sensor_21: float = Field(default=23.32, ge=15.0, le=35.0, description="LPT Coolant Bleed (W32)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "engine_id": 1,
                "cycle": 145,
                "setting_1": -0.0007,
                "setting_2": -0.0004,
                "setting_3": 100.0,
                "sensor_1": 518.67,
                "sensor_2": 642.86,
                "sensor_3": 1592.55,
                "sensor_4": 1410.22,
                "sensor_5": 14.62,
                "sensor_6": 21.61,
                "sensor_7": 553.12,
                "sensor_8": 2388.11,
                "sensor_9": 9054.33,
                "sensor_10": 1.3,
                "sensor_11": 47.65,
                "sensor_12": 521.28,
                "sensor_13": 2388.08,
                "sensor_14": 8130.45,
                "sensor_15": 8.44,
                "sensor_16": 0.03,
                "sensor_17": 393.0,
                "sensor_18": 2388.0,
                "sensor_19": 100.0,
                "sensor_20": 38.72,
                "sensor_21": 23.28
            }
        }
    }


class PredictionRequest(BaseModel):
    """Prediction request for RUL and failure probability."""
    engine_id: int = Field(..., ge=1)
    telemetry: SensorTelemetryIn
    model_type: Optional[str] = Field(
        default="xgboost",
        description="Model architecture to query: 'xgboost' (tabular baseline) or 'lstm' (temporal sequence)"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    batch: List[SensorTelemetryIn]
    model_type: Optional[str] = "xgboost"


class TopFeatureContribution(BaseModel):
    """Individual feature contribution to risk score (e.g. from SHAP)."""
    feature_name: str
    sensor_description: str
    impact_direction: str  # 'increases_risk' or 'decreases_risk'
    importance_weight: float  # Percentage contribution


class PredictionResponse(BaseModel):
    """Response payload for machine RUL and failure risk prediction."""
    engine_id: int
    cycle: int
    predicted_rul: float = Field(..., description="Estimated Remaining Useful Life in cycles")
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated failure probability (0 to 1)")
    health_status: str = Field(..., description="'HEALTHY', 'WARNING', or 'CRITICAL'")
    model_used: str
    top_risk_drivers: List[TopFeatureContribution] = []
    recommended_action: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    """Response payload for batch predictions."""
    predictions: List[PredictionResponse]
    total_processed: int
    total_latency_ms: float


class TelemetryResponse(BaseModel):
    """Response payload after ingesting live telemetry."""
    status: str
    engine_id: int
    cycle: int
    prediction: PredictionResponse
    message: str


class HealthCheckResponse(BaseModel):
    """System health check payload."""
    status: str
    app_name: str
    version: str
    database_connected: bool
    redis_connected: bool
    models_loaded: Dict[str, bool]
    environment: str


class ModelVersionResponse(BaseModel):
    """Model metadata response."""
    production_model: str
    xgboost_version: str
    lstm_version: str
    training_dataset: str
    evaluation_metrics: Dict[str, float]
