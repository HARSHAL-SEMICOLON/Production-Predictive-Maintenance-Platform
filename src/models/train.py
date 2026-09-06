"""Unified Training Pipeline for Sentinel AI with MLflow Experiment Tracking."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import mlflow
import numpy as np
import pandas as pd

from src.config import settings
from src.utils.logger import logger
from src.ingestion.cmapss_loader import CMAPSSDataLoader
from src.validation.data_validator import DataValidator
from src.features.engineering import FeatureEngineeringPipeline
from src.models.baseline_xgb import XGBoostRULModel
from src.models.deep_lstm import LSTMRULModel
from src.models.evaluate import evaluate_rul_predictions


def train_models(
    dataset_name: str = "FD001",
    lstm_epochs: int = 25,
    quick_run: bool = False,
) -> Dict[str, Any]:
    """Orchestrates end-to-end data preparation, model training, MLflow logging, and evaluation."""
    logger.info(f"=== Starting Sentinel AI Training Pipeline for {dataset_name} ===")

    # 1. Setup MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # 2. Data Ingestion
    loader = CMAPSSDataLoader()
    train_df, test_df, rul_df = loader.load_or_generate_data(sub_dataset=dataset_name)

    # 3. Data Validation
    val_report = DataValidator.validate_batch_dataframe(train_df)
    if not val_report["is_valid"]:
        logger.warning(f"Data validation warnings: {val_report['validation_messages']}")

    # 4. Feature Engineering
    fe_pipeline = FeatureEngineeringPipeline(window_size=settings.SEQUENCE_WINDOW_SIZE)
    train_feat_df, feature_cols = fe_pipeline.fit_transform_tabular(train_df)
    test_feat_df = fe_pipeline.transform_tabular(test_df)

    # Prepare Tabular Data for XGBoost
    X_train_tab = train_feat_df[feature_cols]
    y_train_tab = train_feat_df["RUL_clipped"].values

    # For testing tabular: evaluate on the last cycle of each test engine
    test_last_cycles = test_feat_df.groupby("engine_id").last().reset_index()
    X_test_last_tab = test_last_cycles[feature_cols]
    y_test_last = test_last_cycles["RUL_clipped"].values

    # Prepare Sequences for LSTM
    X_train_seq, y_train_seq, _ = fe_pipeline.create_lstm_sequences(
        train_feat_df, feature_cols=settings.INFORMATIVE_SENSOR_COLS
    )

    # Prepare Test Sequences for LSTM: take last sequence for each engine
    test_sequences = []
    test_seq_ruls = []
    for engine_id, group in test_feat_df.groupby("engine_id"):
        seq, ruls, _ = fe_pipeline.create_lstm_sequences(
            group, feature_cols=settings.INFORMATIVE_SENSOR_COLS
        )
        if len(seq) > 0:
            test_sequences.append(seq[-1])
            test_seq_ruls.append(ruls[-1])
    X_test_seq = np.array(test_sequences, dtype=np.float32)
    y_test_seq = np.array(test_seq_ruls, dtype=np.float32)

    # If quick_run is specified (for tests / CI), use fewer trees and epochs
    xgb_trees = 30 if quick_run else 150
    lstm_epochs = 3 if quick_run else lstm_epochs

    results = {}

    # -------------------------------------------------------------
    # Model 1: XGBoost Baseline
    # -------------------------------------------------------------
    logger.info("--- Training Model 1: XGBoost Regressor ---")
    with mlflow.start_run(run_name=f"XGBoost_Baseline_{dataset_name}"):
        mlflow.log_param("model_type", "XGBoostRegressor")
        mlflow.log_param("n_estimators", xgb_trees)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("window_size", settings.SEQUENCE_WINDOW_SIZE)
        mlflow.log_param("rul_clip_threshold", settings.RUL_CLIP_THRESHOLD)
        mlflow.log_param("features_count", len(feature_cols))

        xgb_model = XGBoostRULModel(n_estimators=xgb_trees, max_depth=6, learning_rate=0.05)
        xgb_model.fit(X_train_tab, y_train_tab, feature_names=feature_cols)

        # Predictions & Evaluation
        xgb_test_preds = xgb_model.predict(X_test_last_tab)
        xgb_metrics = evaluate_rul_predictions(y_test_last, xgb_test_preds)

        for k, v in xgb_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        xgb_model.save()
        results["xgboost"] = {
            "metrics": xgb_metrics,
            "params": xgb_model.params,
        }
        logger.info(f"XGBoost Test Metrics: {xgb_metrics}")

    # -------------------------------------------------------------
    # Model 2: Deep Bidirectional LSTM
    # -------------------------------------------------------------
    logger.info("--- Training Model 2: Deep Bidirectional LSTM ---")
    with mlflow.start_run(run_name=f"BiLSTM_Deep_{dataset_name}"):
        mlflow.log_param("model_type", "Bidirectional_LSTM")
        mlflow.log_param("epochs", lstm_epochs)
        mlflow.log_param("lstm_units", 64)
        mlflow.log_param("sequence_length", settings.SEQUENCE_WINDOW_SIZE)
        mlflow.log_param("num_features", X_train_seq.shape[2])

        lstm_model = LSTMRULModel(
            sequence_length=settings.SEQUENCE_WINDOW_SIZE,
            num_features=X_train_seq.shape[2],
            lstm_units=64,
            dropout_rate=0.2,
            learning_rate=0.001,
        )
        lstm_model.build_model()
        lstm_model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_test_seq, y_test_seq) if len(X_test_seq) > 0 else None,
            epochs=lstm_epochs,
            batch_size=64,
        )

        # Predictions & Evaluation
        lstm_test_preds = lstm_model.predict(X_test_seq)
        lstm_metrics = evaluate_rul_predictions(y_test_seq, lstm_test_preds)

        for k, v in lstm_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        lstm_model.save()
        results["lstm"] = {
            "metrics": lstm_metrics,
        }
        logger.info(f"LSTM Test Metrics: {lstm_metrics}")

    # Determine Champion Model based on test RMSE
    xgb_rmse = results["xgboost"]["metrics"]["rmse"]
    lstm_rmse = results["lstm"]["metrics"]["rmse"]
    champion = "xgboost" if xgb_rmse <= lstm_rmse else "lstm"

    production_meta = {
        "champion_model": champion,
        "dataset": dataset_name,
        "xgboost_metrics": results["xgboost"]["metrics"],
        "lstm_metrics": results["lstm"]["metrics"],
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    meta_file = settings.MODELS_DIR / "production_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(production_meta, f, indent=2)

    logger.info(
        f"=== Model Comparison Summary ===\n"
        f"XGBoost RMSE: {xgb_rmse:.2f} | MAE: {results['xgboost']['metrics']['mae']:.2f} | NASA Score: {results['xgboost']['metrics']['nasa_asymmetric_score']:.1f}\n"
        f"LSTM    RMSE: {lstm_rmse:.2f} | MAE: {results['lstm']['metrics']['mae']:.2f} | NASA Score: {results['lstm']['metrics']['nasa_asymmetric_score']:.1f}\n"
        f"Champion Model: {champion.upper()} -> Metadata written to {meta_file}"
    )

    return production_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentinel AI Predictive Maintenance Models")
    parser.add_argument("--dataset", type=str, default="FD001", help="C-MAPSS sub-dataset (e.g. FD001)")
    parser.add_argument("--epochs", type=int, default=25, help="LSTM training epochs")
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced epochs and trees")
    args = parser.parse_args()

    train_models(dataset_name=args.dataset, lstm_epochs=args.epochs, quick_run=args.quick)
