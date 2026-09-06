"""Feature Engineering Pipeline for Sentinel AI Predictive Maintenance.

Builds temporal features, rolling statistics, rate-of-change, and generates:
1. 2D Tabular Feature Matrix for XGBoost Regressor
2. 3D Sequence Tensors (samples, sequence_length=30, features) for LSTM
"""

from typing import List, Tuple, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import settings
from src.utils.logger import logger


class FeatureEngineeringPipeline:
    """Computes time-series rolling aggregations, lag features, and sequence arrays."""

    def __init__(
        self,
        window_size: int = settings.SEQUENCE_WINDOW_SIZE,
        scaler_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.scaler_path = scaler_path or (settings.MODELS_DIR / "scaler.joblib")
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_columns: List[str] = []

    def fit_transform_tabular(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fits MinMaxScaler and generates tabular features on training data."""
        featured_df = self._create_time_series_features(df)
        feature_cols = [c for c in featured_df.columns if c not in ["engine_id", "cycle", "RUL", "RUL_clipped", "is_critical"]]
        self.feature_columns = feature_cols

        self.scaler = MinMaxScaler()
        featured_df[feature_cols] = self.scaler.fit_transform(featured_df[feature_cols])

        # Persist scaler
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Fitted MinMaxScaler on {len(feature_cols)} features and saved to {self.scaler_path}")

        return featured_df, feature_cols

    def transform_tabular(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms unseen data using previously fitted scaler and feature definitions."""
        featured_df = self._create_time_series_features(df)
        if self.scaler is None:
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            else:
                raise ValueError("Scaler not fitted and no saved scaler found at {self.scaler_path}")

        if not self.feature_columns:
            self.feature_columns = [
                c for c in featured_df.columns if c not in ["engine_id", "cycle", "RUL", "RUL_clipped", "is_critical"]
            ]

        # Ensure all columns exist
        for col in self.feature_columns:
            if col not in featured_df.columns:
                featured_df[col] = 0.0

        featured_df[self.feature_columns] = self.scaler.transform(featured_df[self.feature_columns])
        return featured_df

    def _create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling statistics, rate of change, and lag features per engine."""
        df = df.copy()
        df.sort_values(by=["engine_id", "cycle"], inplace=True)

        sensor_cols = settings.INFORMATIVE_SENSOR_COLS
        rolling_windows = [5, 10, 20]

        # 1. Rolling Mean and Rolling Standard Deviation
        for w in rolling_windows:
            roll_mean = (
                df.groupby("engine_id")[sensor_cols]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            roll_mean.columns = [f"{c}_roll_mean_{w}" for c in sensor_cols]

            roll_std = (
                df.groupby("engine_id")[sensor_cols]
                .rolling(window=w, min_periods=1)
                .std()
                .fillna(0.0)
                .reset_index(level=0, drop=True)
            )
            roll_std.columns = [f"{c}_roll_std_{w}" for c in sensor_cols]

            df = pd.concat([df, roll_mean, roll_std], axis=1)

        # 2. Lag Features (values at t-1, t-3, t-5)
        for lag in [1, 3, 5]:
            lag_df = (
                df.groupby("engine_id")[sensor_cols]
                .shift(lag)
                .bfill()
            )
            lag_df.columns = [f"{c}_lag_{lag}" for c in sensor_cols]
            df = pd.concat([df, lag_df], axis=1)

        # 3. Rate of Change (1st difference Delta = Current - Previous)
        for c in sensor_cols:
            df[f"{c}_roc"] = df.groupby("engine_id")[c].diff().fillna(0.0)

        # 4. Engine Health Degradation Index
        # Compare current sensor reading against engine's initial baseline average (first 5 cycles)
        for c in ["sensor_2", "sensor_3", "sensor_4", "sensor_11", "sensor_15"]:
            if c in df.columns:
                initial_baseline = (
                    df[df["cycle"] <= 5].groupby("engine_id")[c].transform("mean")
                )
                df[f"{c}_deviation_from_baseline"] = df[c] - initial_baseline

        return df

    def create_lstm_sequences(
        self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates 3D sequential arrays for LSTM: (samples, window_size=30, features).
        
        Returns:
            X_seq: 3D numpy array
            y_rul: 1D Remaining Useful Life targets
            engine_ids: 1D engine identifiers
        """
        if feature_cols is None:
            feature_cols = self.feature_columns or settings.INFORMATIVE_SENSOR_COLS

        X_sequences = []
        y_ruls = []
        engine_id_list = []

        for engine_id, group in df.groupby("engine_id"):
            group = group.sort_values(by="cycle")
            feature_matrix = group[feature_cols].values
            if "RUL_clipped" in group.columns:
                rul_values = group["RUL_clipped"].values
            elif "RUL" in group.columns:
                rul_values = group["RUL"].values
            else:
                rul_values = np.zeros(len(group), dtype=np.float32)

            num_cycles = len(group)

            if num_cycles < self.window_size:
                # Pad sequences if shorter than window_size
                pad_length = self.window_size - num_cycles
                padding = np.repeat(feature_matrix[:1], pad_length, axis=0)
                padded_matrix = np.vstack([padding, feature_matrix])
                X_sequences.append(padded_matrix)
                y_ruls.append(rul_values[-1])
                engine_id_list.append(engine_id)
            else:
                for i in range(self.window_size, num_cycles + 1):
                    X_sequences.append(feature_matrix[i - self.window_size : i])
                    y_ruls.append(rul_values[i - 1])
                    engine_id_list.append(engine_id)

        return np.array(X_sequences, dtype=np.float32), np.array(y_ruls, dtype=np.float32), np.array(engine_id_list)
