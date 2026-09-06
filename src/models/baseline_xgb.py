"""XGBoost Regressor baseline model for Remaining Useful Life (RUL) with SHAP Explainability."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import settings
from src.utils.logger import logger


class XGBoostRULModel:
    """Production XGBoost model for Remaining Useful Life (RUL) prediction and SHAP attribution."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": kwargs.get("objective", "reg:squarederror"),
            "n_jobs": kwargs.get("n_jobs", -1),
        }
        self.params.update(kwargs)
        self.model = XGBRegressor(**self.params)
        self.feature_names: List[str] = []
        self.explainer: Optional[Any] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fits XGBoost Regressor on tabular feature matrix."""
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"f_{i}" for i in range(X_train.shape[1])]

        logger.info(f"Training XGBoost Regressor on {X_train.shape[0]} samples with {len(self.feature_names)} features...")
        self.model.fit(X_train, y_train)
        logger.info("XGBoost training complete.")

        # Initialize TreeExplainer for SHAP explanations
        try:
            import shap
            # Use background summary for efficient TreeExplainer
            sample_bg = shap.sample(X_train, min(100, len(X_train)), random_state=42)
            self.explainer = shap.TreeExplainer(self.model, data=sample_bg)
            logger.info("Initialized SHAP TreeExplainer.")
        except Exception as e:
            logger.warning(f"SHAP explainer initialization skipped: {e}")
            self.explainer = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts Remaining Useful Life in cycles."""
        preds = self.model.predict(X)
        # Physical lower bound: RUL cannot be negative
        return np.maximum(preds, 0.0)

    def calculate_failure_probability(self, predicted_rul: float) -> float:
        """Converts estimated RUL into a calibrated Failure Probability (0.0 to 1.0).
        
        Uses a smooth logistic sigmoid function centered around the critical threshold:
        P(failure) = 1 / (1 + exp((RUL - threshold) / scale))
        """
        threshold = settings.CRITICAL_RUL_THRESHOLD  # e.g., 30 cycles
        scale = 12.0  # Transition smoothness
        # Logistic curve: when RUL=30, P=0.5; when RUL=10, P=0.84; when RUL=5, P=0.91; when RUL=80, P=0.01
        prob = 1.0 / (1.0 + np.exp((predicted_rul - threshold) / scale))
        return float(np.clip(prob, 0.001, 0.999))

    def get_health_status(self, predicted_rul: float) -> Tuple[str, str]:
        """Categorizes asset status and yields operational recommendation."""
        if predicted_rul <= settings.CRITICAL_RUL_THRESHOLD:
            return "CRITICAL", "Immediate shutdown and scheduled component overhaul required."
        elif predicted_rul <= settings.WARNING_RUL_THRESHOLD:
            return "WARNING", "Elevated degradation detected. Schedule maintenance inspection within 24 hours."
        else:
            return "HEALTHY", "Normal operating conditions. Continue nominal operational cycle."

    def explain_prediction(self, X_single: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """Calculates top feature contributions to risk using SHAP or feature importances."""
        results = []
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X_single)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                values = shap_values[0] if len(shap_values.shape) > 1 else shap_values

                # Higher negative SHAP on RUL means that feature drives down remaining cycles (elevates risk)
                # We sort by absolute magnitude
                indices = np.argsort(np.abs(values))[::-1][:top_k]
                total_mag = np.sum(np.abs(values)) + 1e-8

                for idx in indices:
                    feat = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                    val = float(values[idx])
                    # In RUL regression: negative SHAP reduces RUL -> increases failure risk
                    impact = "increases_risk" if val < 0 else "decreases_risk"
                    weight = round((abs(val) / total_mag) * 100, 1)

                    # Extract root sensor name for human-readable description
                    root_sensor = feat.split("_")[0] + "_" + feat.split("_")[1] if "sensor_" in feat else feat
                    desc = settings.SENSOR_DESCRIPTIONS.get(root_sensor, feat)

                    results.append({
                        "feature_name": feat,
                        "sensor_description": desc,
                        "impact_direction": impact,
                        "importance_weight": weight,
                    })
                return results
            except Exception as e:
                logger.debug(f"SHAP explanation computation failed: {e}")

        # Fallback to model feature importances if SHAP fails
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_k]
        total_imp = np.sum(importances) + 1e-8
        for idx in top_indices:
            feat = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            root_sensor = feat.split("_")[0] + "_" + feat.split("_")[1] if "sensor_" in feat else feat
            desc = settings.SENSOR_DESCRIPTIONS.get(root_sensor, feat)
            results.append({
                "feature_name": feat,
                "sensor_description": desc,
                "impact_direction": "increases_risk",
                "importance_weight": round(float(importances[idx] / total_imp) * 100, 1),
            })
        return results

    def save(self, filepath: Optional[Path] = None):
        """Saves model and metadata to disk."""
        filepath = filepath or (settings.MODELS_DIR / "xgboost_model.joblib")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
        }, filepath)
        logger.info(f"Saved XGBoost model to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "XGBoostRULModel":
        """Loads model and metadata from disk."""
        filepath = filepath or (settings.MODELS_DIR / "xgboost_model.joblib")
        if not filepath.exists():
            raise FileNotFoundError(f"No XGBoost model found at {filepath}")
        data = joblib.load(filepath)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance.feature_names = data.get("feature_names", [])
        logger.info(f"Loaded XGBoost model from {filepath}")
        return instance
