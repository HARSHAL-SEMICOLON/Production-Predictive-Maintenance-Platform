"""Evaluation metrics suite including the official NASA C-MAPSS Asymmetric Score."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the official NASA C-MAPSS asymmetric evaluation score.

    Mathematical formulation:
    d_i = y_pred_i - y_true_i
    
    If d_i < 0 (early prediction, maintenance scheduled before failure):
        s_i = exp(-d_i / 13) - 1
    If d_i >= 0 (late prediction, catastrophic failure occurred unexpectedly):
        s_i = exp(d_i / 10) - 1

    Total Score S = sum(s_i)
    
    Interview Defense:
    In industrial manufacturing and aviation, asymmetric loss reflects true economic
    risk. A late prediction causes unplanned downtime and potential catastrophic loss,
    so it is penalized exponentially harsher (divisor 10) than an early prediction (divisor 13).
    """
    diff = y_pred - y_true
    scores = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(scores))


def evaluate_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression metrics for predictive maintenance."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    nasa_score = compute_nasa_scoring_function(y_true, y_pred)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "nasa_asymmetric_score": round(nasa_score, 2),
    }
