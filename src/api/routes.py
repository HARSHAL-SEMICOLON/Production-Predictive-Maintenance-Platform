"""REST API Endpoints for Sentinel AI Predictive Maintenance Platform."""

import time
import json
from typing import List, Optional
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session

from src.config import settings
from src.utils.logger import logger
from src.validation.schemas import (
    SensorTelemetryIn,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    TelemetryResponse,
    HealthCheckResponse,
    ModelVersionResponse,
    TopFeatureContribution,
)
from src.validation.data_validator import DataValidator
from src.features.engineering import FeatureEngineeringPipeline
from src.models.baseline_xgb import XGBoostRULModel
from src.models.deep_lstm import LSTMRULModel
from src.models.retrain import trigger_automated_retraining
from src.monitoring.drift_detector import drift_detector
from src.monitoring.prometheus_exporter import (
    PREDICTION_REQUESTS,
    PREDICTION_LATENCY,
    TELEMETRY_INGEST_COUNT,
    FAILURE_ALERTS_TOTAL,
    PREDICTED_RUL_GAUGE,
    FAILURE_RISK_GAUGE,
)
from src.api.database import get_db, SensorReading, PredictionLog, AlertLog, ModelRun, engine as db_engine
from src.api.cache import cache

router = APIRouter()

# Global model container (initialized on startup in main.py)
model_registry = {
    "xgboost": None,
    "lstm": None,
    "pipeline": None,
    "champion": "xgboost",
    "production_meta": {},
}


def get_feature_pipeline() -> FeatureEngineeringPipeline:
    if model_registry["pipeline"] is None:
        model_registry["pipeline"] = FeatureEngineeringPipeline()
    return model_registry["pipeline"]


@router.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check():
    """Liveness and readiness health probe."""
    db_ok = True
    try:
        with db_engine.connect() as conn:
            pass
    except Exception:
        db_ok = False

    redis_ok = cache.is_connected() or cache.use_memory_fallback

    return HealthCheckResponse(
        status="healthy" if (db_ok and (model_registry["xgboost"] is not None or model_registry["lstm"] is not None)) else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        database_connected=db_ok,
        redis_connected=redis_ok,
        models_loaded={
            "xgboost": model_registry["xgboost"] is not None,
            "lstm": model_registry["lstm"] is not None,
        },
        environment=settings.ENVIRONMENT,
    )


@router.get("/version", response_model=ModelVersionResponse, tags=["System"])
def get_model_version():
    """Returns metadata regarding active models and performance metrics."""
    meta = model_registry.get("production_meta", {})
    champion = meta.get("champion_model", model_registry["champion"])
    metrics = meta.get(f"{champion}_metrics", {})

    return ModelVersionResponse(
        production_model=champion.upper(),
        xgboost_version="3.0.5",
        lstm_version="Keras 3.9 / TF 2.19",
        training_dataset=meta.get("dataset", "NASA C-MAPSS FD001"),
        evaluation_metrics=metrics,
    )


@router.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_rul(request: PredictionRequest, db: Session = Depends(get_db)):
    """Estimates Remaining Useful Life (RUL) and failure probability from sensor data."""
    start_time = time.time()
    model_type = (request.model_type or model_registry["champion"]).lower()

    # Telemetry data dictionary
    telemetry_dict = request.telemetry.model_dump()
    engine_id = request.engine_id
    cycle = request.telemetry.cycle

    # 1. Feature Engineering
    pipeline = get_feature_pipeline()
    df_raw = pd.DataFrame([telemetry_dict])

    try:
        df_feat = pipeline.transform_tabular(df_raw)
    except Exception:
        # If pipeline not fitted, fit dummy baseline
        df_feat, _ = pipeline.fit_transform_tabular(df_raw)

    feat_cols = [c for c in df_feat.columns if c not in ["engine_id", "cycle", "RUL", "RUL_clipped", "is_critical"]]

    # 2. Model Inference
    if model_type == "lstm" and model_registry["lstm"] is not None:
        # For LSTM: get 30-cycle sequence from cache
        readings = cache.get_recent_readings(engine_id, count=settings.SEQUENCE_WINDOW_SIZE)
        if len(readings) < settings.SEQUENCE_WINDOW_SIZE:
            # Pad with current reading
            while len(readings) < settings.SEQUENCE_WINDOW_SIZE:
                readings.insert(0, telemetry_dict)

        seq_df = pd.DataFrame(readings)
        seq_array, _, _ = pipeline.create_lstm_sequences(seq_df, feature_cols=settings.INFORMATIVE_SENSOR_COLS)
        if len(seq_array) > 0:
            rul_pred = float(model_registry["lstm"].predict(seq_array[-1:])[0])
        else:
            rul_pred = 80.0
        used_model = "Bidirectional LSTM"
        top_drivers = []
    else:
        # XGBoost Baseline
        xgb: Optional[XGBoostRULModel] = model_registry["xgboost"]
        if xgb is None:
            raise HTTPException(status_code=503, detail="XGBoost model is not loaded yet.")

        # Match columns expected by model
        expected_cols = xgb.feature_names or feat_cols
        for c in expected_cols:
            if c not in df_feat.columns:
                df_feat[c] = 0.0

        X_input = df_feat[expected_cols]
        rul_pred = float(xgb.predict(X_input)[0])
        used_model = "XGBoost Regressor"
        
        # Explainability via SHAP / feature importances
        raw_drivers = xgb.explain_prediction(X_input, top_k=5)
        top_drivers = [TopFeatureContribution(**d) for d in raw_drivers]

    latency = round((time.time() - start_time) * 1000, 2)

    # Derive probability and health categorization
    xgb_inst: XGBoostRULModel = model_registry["xgboost"]
    fail_prob = xgb_inst.calculate_failure_probability(rul_pred) if xgb_inst else 0.5
    status_str, action_str = xgb_inst.get_health_status(rul_pred) if xgb_inst else ("HEALTHY", "Normal")

    # Update Prometheus Metrics
    PREDICTION_REQUESTS.labels(model_type=used_model, status="success").inc()
    PREDICTION_LATENCY.labels(model_type=used_model).observe(latency / 1000.0)
    PREDICTED_RUL_GAUGE.labels(engine_id=str(engine_id)).set(rul_pred)
    FAILURE_RISK_GAUGE.labels(engine_id=str(engine_id)).set(fail_prob)

    # Persist prediction in DB
    pred_log = PredictionLog(
        engine_id=engine_id,
        cycle=cycle,
        predicted_rul=rul_pred,
        failure_probability=fail_prob,
        health_status=status_str,
        model_used=used_model,
        latency_ms=latency,
        top_drivers=json.dumps([d.model_dump() for d in top_drivers]),
    )
    db.add(pred_log)
    db.commit()

    return PredictionResponse(
        engine_id=engine_id,
        cycle=cycle,
        predicted_rul=round(rul_pred, 1),
        failure_probability=round(fail_prob, 3),
        health_status=status_str,
        model_used=used_model,
        top_risk_drivers=top_drivers,
        recommended_action=action_str,
        latency_ms=latency,
    )


@router.post("/api/v1/telemetry", response_model=TelemetryResponse, tags=["Ingestion"])
def ingest_telemetry(telemetry: SensorTelemetryIn, db: Session = Depends(get_db)):
    """Ingests live IoT edge telemetry, validates data quality, and computes instantaneous risk."""
    telemetry_data = telemetry.model_dump()
    engine_id = telemetry.engine_id
    cycle = telemetry.cycle

    # 1. Validation
    is_valid, validation_errors = DataValidator.validate_telemetry_dict(telemetry_data)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Telemetry validation failed", "errors": validation_errors},
        )

    TELEMETRY_INGEST_COUNT.inc()

    # 2. Push to Redis/Memory sliding cache
    cache.push_reading(engine_id, telemetry_data)

    # 3. Store raw reading in Database
    reading_record = SensorReading(**telemetry_data)
    db.add(reading_record)
    db.commit()

    # 4. Predict RUL using champion model
    pred_req = PredictionRequest(
        engine_id=engine_id,
        telemetry=telemetry,
        model_type=model_registry["champion"],
    )
    prediction = predict_rul(pred_req, db=db)

    # 5. Check if alert should be triggered
    if prediction.health_status in ["WARNING", "CRITICAL"]:
        FAILURE_ALERTS_TOTAL.labels(alert_level=prediction.health_status).inc()
        alert = AlertLog(
            engine_id=engine_id,
            cycle=cycle,
            alert_level=prediction.health_status,
            message=f"Engine {engine_id} at cycle {cycle}: {prediction.recommended_action} (RUL: {prediction.predicted_rul})",
        )
        db.add(alert)
        db.commit()

    return TelemetryResponse(
        status="success",
        engine_id=engine_id,
        cycle=cycle,
        prediction=prediction,
        message="Telemetry ingested and verified.",
    )


@router.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_req: BatchPredictionRequest, db: Session = Depends(get_db)):
    """High-throughput batch prediction for fleet monitoring."""
    start_time = time.time()
    predictions = []

    for item in batch_req.batch:
        req = PredictionRequest(engine_id=item.engine_id, telemetry=item, model_type=batch_req.model_type)
        pred = predict_rul(req, db=db)
        predictions.append(pred)

    total_latency = round((time.time() - start_time) * 1000, 2)
    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        total_latency_ms=total_latency,
    )


@router.get("/api/v1/telemetry/recent", tags=["Telemetry"])
def get_recent_telemetry(engine_id: int = 1, limit: int = 30, db: Session = Depends(get_db)):
    """Fetches recent sensor history and predictions for dashboard charts."""
    readings = (
        db.query(SensorReading)
        .filter(SensorReading.engine_id == engine_id)
        .order_by(SensorReading.cycle.desc())
        .limit(limit)
        .all()
    )
    # Reverse to chronological order
    readings = readings[::-1]

    preds = (
        db.query(PredictionLog)
        .filter(PredictionLog.engine_id == engine_id)
        .order_by(PredictionLog.cycle.desc())
        .limit(limit)
        .all()
    )[::-1]

    return {
        "engine_id": engine_id,
        "count": len(readings),
        "readings": [
            {
                "cycle": r.cycle,
                "sensor_2": r.sensor_2,
                "sensor_3": r.sensor_3,
                "sensor_4": r.sensor_4,
                "sensor_8": r.sensor_8,
                "sensor_11": r.sensor_11,
                "sensor_13": r.sensor_13,
                "sensor_15": r.sensor_15,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in readings
        ],
        "predictions": [
            {
                "cycle": p.cycle,
                "predicted_rul": p.predicted_rul,
                "failure_probability": p.failure_probability,
                "health_status": p.health_status,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in preds
        ],
    }


@router.get("/api/v1/alerts", tags=["Alerts"])
def get_alerts(limit: int = 20, db: Session = Depends(get_db)):
    """Retrieves recent maintenance alerts and critical anomalies."""
    alerts = (
        db.query(AlertLog)
        .order_by(AlertLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": a.id,
            "engine_id": a.engine_id,
            "cycle": a.cycle,
            "alert_level": a.alert_level,
            "message": a.message,
            "acknowledged": a.acknowledged,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in alerts
    ]


@router.get("/api/v1/drift/report", tags=["Monitoring"])
def get_drift_report(db: Session = Depends(get_db)):
    """Executes Evidently AI statistical drift detection between baseline and production telemetry."""
    # Query recent 200 readings from DB
    readings = db.query(SensorReading).order_by(SensorReading.id.desc()).limit(200).all()
    if len(readings) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 10 logged telemetry readings to analyze drift; found {len(readings)}.",
        }

    data_dicts = []
    for r in readings:
        d = {f"sensor_{i}": getattr(r, f"sensor_{i}") for i in range(1, 22)}
        d["engine_id"] = r.engine_id
        d["cycle"] = r.cycle
        data_dicts.append(d)

    df_prod = pd.DataFrame(data_dicts)
    summary = drift_detector.run_drift_analysis(df_prod)
    return summary


@router.get("/api/v1/drift/report/html", response_class=HTMLResponse, tags=["Monitoring"])
def get_drift_report_html():
    """Serves the interactive HTML data drift report."""
    html_path = settings.REPORTS_DIR / "drift_report.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<html><body><h3>No drift report generated yet. Run /api/v1/drift/report first.</h3></body></html>",
            status_code=404,
        )
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.post("/api/v1/retrain", tags=["MLOps"])
def trigger_retrain(background_tasks: BackgroundTasks):
    """Triggers background automated retraining."""
    background_tasks.add_task(trigger_automated_retraining, "Manual API Trigger")
    return {
        "status": "retraining_scheduled",
        "message": "Model retraining initiated in background. MLflow tracking will record experiment.",
    }
