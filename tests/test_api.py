"""Integration tests for Sentinel AI FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import model_registry
from src.models.baseline_xgb import XGBoostRULModel
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def client():
    # Ensure a basic model is in model_registry for testing
    if model_registry["xgboost"] is None:
        xgb = XGBoostRULModel(n_estimators=5, max_depth=2)
        X = pd.DataFrame(np.random.randn(20, 5), columns=[f"f_{i}" for i in range(5)])
        y = np.linspace(100, 10, 20)
        xgb.fit(X, y)
        model_registry["xgboost"] = xgb
        model_registry["champion"] = "xgboost"

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "app_name" in data
    assert "version" in data


def test_version_endpoint(client):
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "production_model" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "sentinel_prediction_requests_total" in response.text


def test_telemetry_ingestion_and_prediction(client):
    payload = {
        "engine_id": 1,
        "cycle": 15,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": 642.50,
        "sensor_3": 1588.0,
        "sensor_4": 1404.0,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 553.8,
        "sensor_8": 2388.08,
        "sensor_9": 9055.0,
        "sensor_10": 1.30,
        "sensor_11": 47.45,
        "sensor_12": 521.8,
        "sensor_13": 2388.09,
        "sensor_14": 8135.0,
        "sensor_15": 8.42,
        "sensor_16": 0.03,
        "sensor_17": 392.5,
        "sensor_18": 2388.0,
        "sensor_19": 100.0,
        "sensor_20": 38.85,
        "sensor_21": 23.32,
    }
    response = client.post("/api/v1/telemetry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["engine_id"] == 1
    assert "prediction" in data
    assert "predicted_rul" in data["prediction"]
    assert "failure_probability" in data["prediction"]
