"""Prometheus Metrics Instrumentation for Sentinel AI."""

from prometheus_client import Counter, Histogram, Gauge

# Counters
PREDICTION_REQUESTS = Counter(
    "sentinel_prediction_requests_total",
    "Total count of inference requests processed",
    ["model_type", "status"],
)

TELEMETRY_INGEST_COUNT = Counter(
    "sentinel_telemetry_ingested_total",
    "Total real-time IoT telemetry readings received",
)

FAILURE_ALERTS_TOTAL = Counter(
    "sentinel_failure_alerts_total",
    "Total count of predictive maintenance warnings and alerts triggered",
    ["alert_level"],
)

# Histograms
PREDICTION_LATENCY = Histogram(
    "sentinel_prediction_latency_seconds",
    "End-to-end model inference latency in seconds",
    ["model_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Gauges
PREDICTED_RUL_GAUGE = Gauge(
    "sentinel_predicted_rul_cycles",
    "Latest predicted Remaining Useful Life in operational cycles",
    ["engine_id"],
)

FAILURE_RISK_GAUGE = Gauge(
    "sentinel_failure_risk_probability",
    "Calibrated failure probability (0.0 to 1.0)",
    ["engine_id"],
)

DATA_DRIFT_SCORE_GAUGE = Gauge(
    "sentinel_data_drift_score",
    "Latest dataset drift share percentage calculated by Evidently AI",
)
