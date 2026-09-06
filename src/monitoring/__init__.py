"""Monitoring module."""
from src.monitoring.prometheus_exporter import (
    PREDICTION_REQUESTS,
    TELEMETRY_INGEST_COUNT,
    FAILURE_ALERTS_TOTAL,
    PREDICTION_LATENCY,
    PREDICTED_RUL_GAUGE,
    FAILURE_RISK_GAUGE,
    DATA_DRIFT_SCORE_GAUGE,
)
from src.monitoring.drift_detector import DriftDetector, drift_detector

__all__ = [
    "PREDICTION_REQUESTS",
    "TELEMETRY_INGEST_COUNT",
    "FAILURE_ALERTS_TOTAL",
    "PREDICTION_LATENCY",
    "PREDICTED_RUL_GAUGE",
    "FAILURE_RISK_GAUGE",
    "DATA_DRIFT_SCORE_GAUGE",
    "DriftDetector",
    "drift_detector",
]
