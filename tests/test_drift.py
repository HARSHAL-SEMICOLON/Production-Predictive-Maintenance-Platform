"""Unit tests for Sentinel AI Drift Detection."""

import numpy as np
import pandas as pd
from src.monitoring.drift_detector import DriftDetector
from src.config import settings


def test_drift_detector_clean_data():
    detector = DriftDetector()

    # Generate baseline data
    rows = []
    for _ in range(50):
        row = {}
        for s in settings.INFORMATIVE_SENSOR_COLS:
            row[s] = np.random.normal(100.0, 5.0)
        rows.append(row)
    df_base = pd.DataFrame(rows)
    detector.reference_data = df_base

    # Current data drawn from identical distribution
    rows_curr = []
    for _ in range(50):
        row = {}
        for s in settings.INFORMATIVE_SENSOR_COLS:
            row[s] = np.random.normal(100.0, 5.0)
        rows_curr.append(row)
    df_curr = pd.DataFrame(rows_curr)

    result = detector.run_drift_analysis(df_curr)
    assert result["status"] == "success"
    # Should not detect extreme dataset drift on identical distribution
    assert result["drift_share"] < 0.35


def test_drift_detector_severe_drift():
    detector = DriftDetector()

    # Baseline data
    rows = []
    for _ in range(50):
        row = {}
        for s in settings.INFORMATIVE_SENSOR_COLS:
            row[s] = np.random.normal(100.0, 5.0)
        rows.append(row)
    df_base = pd.DataFrame(rows)
    detector.reference_data = df_base

    # Severely shifted data (+50 units on every sensor)
    rows_shifted = []
    for _ in range(50):
        row = {}
        for s in settings.INFORMATIVE_SENSOR_COLS:
            row[s] = np.random.normal(150.0, 5.0)
        rows_shifted.append(row)
    df_shifted = pd.DataFrame(rows_shifted)

    result = detector.run_drift_analysis(df_shifted)
    assert result["status"] == "success"
    assert result["drift_detected"] is True
    assert result["drift_share"] >= 0.30
