"""Unit tests for Sentinel AI Feature Engineering Pipeline."""

import numpy as np
import pandas as pd
from src.features.engineering import FeatureEngineeringPipeline
from src.config import settings


def test_feature_engineering_tabular():
    # Construct small synthetic trajectory
    rows = []
    for cycle in range(1, 25):
        row = {"engine_id": 1, "cycle": cycle, "RUL": 100 - cycle, "RUL_clipped": min(100 - cycle, 125)}
        for s in settings.INFORMATIVE_SENSOR_COLS:
            row[s] = 100.0 + cycle * 0.5 + np.random.normal(0, 0.1)
        rows.append(row)
    df = pd.DataFrame(rows)

    pipeline = FeatureEngineeringPipeline(window_size=10)
    transformed_df, feature_cols = pipeline.fit_transform_tabular(df)

    assert len(transformed_df) == len(df)
    assert len(feature_cols) > len(settings.INFORMATIVE_SENSOR_COLS)
    # Check rolling feature created
    assert "sensor_2_roll_mean_5" in transformed_df.columns
    assert "sensor_2_roll_std_5" in transformed_df.columns
    assert "sensor_2_lag_1" in transformed_df.columns
    assert "sensor_2_roc" in transformed_df.columns

    # Scaled features should be between 0 and 1
    assert transformed_df[feature_cols].min().min() >= -0.01
    assert transformed_df[feature_cols].max().max() <= 1.01


def test_create_lstm_sequences():
    rows = []
    for cycle in range(1, 40):
        row = {"engine_id": 1, "cycle": cycle, "RUL": 120 - cycle, "RUL_clipped": 100.0}
        for s in settings.INFORMATIVE_SENSOR_COLS:
            row[s] = 10.0 + cycle
        rows.append(row)
    df = pd.DataFrame(rows)

    pipeline = FeatureEngineeringPipeline(window_size=15)
    X_seq, y_rul, engine_ids = pipeline.create_lstm_sequences(df, feature_cols=settings.INFORMATIVE_SENSOR_COLS)

    # Sequence length should be 15
    assert X_seq.shape[1] == 15
    assert X_seq.shape[2] == len(settings.INFORMATIVE_SENSOR_COLS)
    assert len(X_seq) == len(df) - 15 + 1
