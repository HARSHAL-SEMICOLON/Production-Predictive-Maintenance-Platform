"""FastAPI Application Entry Point for Sentinel AI Predictive Maintenance Platform."""

import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import settings
from src.utils.logger import logger
from src.api.database import init_db
from src.api.routes import router, model_registry
from src.features.engineering import FeatureEngineeringPipeline
from src.models.baseline_xgb import XGBoostRULModel
from src.models.deep_lstm import LSTMRULModel
from src.models.train import train_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: database initialization and model loading."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")
    
    # 1. Initialize Database Schema
    init_db()

    # 2. Initialize Feature Engineering Pipeline
    model_registry["pipeline"] = FeatureEngineeringPipeline()

    # 3. Check and Load Models
    xgb_path = settings.MODELS_DIR / "xgboost_model.joblib"
    lstm_path = settings.MODELS_DIR / "lstm_model.keras"
    meta_path = settings.MODELS_DIR / "production_metadata.json"

    if not (xgb_path.exists() and lstm_path.exists()):
        logger.warning("No pre-trained model artifacts found on startup. Executing initial training pipeline...")
        try:
            train_models(dataset_name="FD001", lstm_epochs=3, quick_run=True)
        except Exception as e:
            logger.error(f"Initial model training failed: {e}")

    # Load XGBoost
    try:
        if xgb_path.exists():
            model_registry["xgboost"] = XGBoostRULModel.load(xgb_path)
    except Exception as e:
        logger.error(f"Error loading XGBoost model: {e}")

    # Load LSTM
    try:
        if lstm_path.exists():
            model_registry["lstm"] = LSTMRULModel.load(settings.MODELS_DIR)
    except Exception as e:
        logger.error(f"Error loading LSTM model: {e}")

    # Load production metadata
    if meta_path.exists():
        with open(meta_path, "r") as f:
            model_registry["production_meta"] = json.load(f)
            model_registry["champion"] = model_registry["production_meta"].get("champion_model", "xgboost")

    logger.info(
        f"Startup complete. Champion Model: {model_registry['champion'].upper()} | "
        f"XGBoost: {'Loaded' if model_registry['xgboost'] else 'Not Loaded'} | "
        f"LSTM: {'Loaded' if model_registry['lstm'] else 'Not Loaded'}"
    )

    yield

    logger.info("Shutting down Sentinel AI API...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production Predictive Maintenance Platform for industrial turbofan engines. "
        "Provides real-time Remaining Useful Life (RUL) estimation, failure risk classification, "
        "SHAP feature attributions, and MLOps drift monitoring."
    ),
    lifespan=lifespan,
)

# Configure CORS for Dashboard and external UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Auto-Instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Include API endpoints
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
