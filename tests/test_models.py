"""Unit tests for XGBoost and LSTM models."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline_xgb import XGBoostRULModel
from src.models.deep_lstm import LSTMRULModel
from src.models.evaluate import compute_nasa_scoring_function, evaluate_rul_predictions


def test_xgboost_fit_predict():
    X = pd.DataFrame(np.random.randn(50, 10), columns=[f"f_{i}" for i in range(10)])
    y = np.linspace(100, 10, 50)

    model = XGBoostRULModel(n_estimators=10, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 50
    assert np.all(preds >= 0.0)

    # Test failure probability calibration
    prob_high_rul = model.calculate_failure_probability(120.0)
    prob_low_rul = model.calculate_failure_probability(10.0)

    assert prob_low_rul > prob_high_rul
    assert 0.0 <= prob_high_rul <= 1.0
    assert 0.0 <= prob_low_rul <= 1.0

    # Test health status assignment
    status_crit, action_crit = model.get_health_status(20.0)
    assert status_crit == "CRITICAL"

    status_ok, _ = model.get_health_status(110.0)
    assert status_ok == "HEALTHY"


def test_nasa_scoring_metric():
    y_true = np.array([50.0, 50.0])
    
    # Prediction 1: early by 5 cycles (predicted 45, true 50 => d = -5)
    # Prediction 2: late by 5 cycles (predicted 55, true 50 => d = +5)
    score_early = compute_nasa_scoring_function(np.array([50.0]), np.array([45.0]))
    score_late = compute_nasa_scoring_function(np.array([50.0]), np.array([55.0]))

    # Asymmetric principle: late prediction must be penalized more heavily than early!
    assert score_late > score_early


def test_lstm_model_build_and_predict():
    sequence_len = 10
    num_features = 5
    model = LSTMRULModel(sequence_length=sequence_len, num_features=num_features, lstm_units=16)
    keras_model = model.build_model()
    assert keras_model is not None

    # Dummy inference test
    X_dummy = np.random.randn(4, sequence_len, num_features).astype(np.float32)
    preds = model.predict(X_dummy)
    assert len(preds) == 4
    assert np.all(preds >= 0.0)
