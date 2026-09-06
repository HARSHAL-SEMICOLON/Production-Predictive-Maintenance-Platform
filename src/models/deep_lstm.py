"""Deep Bidirectional LSTM Network for Turbofan Remaining Useful Life (RUL) Prediction.

========================================================================================
ML / MLOps Interview Technical Reference: LSTM Mathematical Formulation
========================================================================================
Standard Recurrent Neural Networks (RNNs) suffer from the Vanishing and Exploding Gradient
problem when backpropagating through time (BPTT) across long horizons (e.g. 30+ engine cycles)
due to continuous matrix multiplications by weight matrices W.

LSTMs solve this via the "Constant Error Carousel" (additive Cell State updates) and
three gating mechanisms:

1. FORGET GATE:
   f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
   Determines what percentage of historical cell state memory to discard (0 = forget, 1 = retain).
   In predictive maintenance, this allows the network to forget irrelevant operational noise
   from ancient healthy flights when recent cycles begin showing wear.

2. INPUT GATE & CANDIDATE STATE:
   i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)
   ~C_t = tanh(W_c * [h_{t-1}, x_t] + b_c)
   Controls what new sensor degradation patterns from current cycle x_t are written into memory.

3. CELL STATE UPDATE (The Highway):
   C_t = f_t * C_{t-1} + i_t * ~C_t
   Notice this update is ADDITIVE. The gradient dC_t / dC_{t-1} contains f_t, preventing
   gradients from vanishing across dozens of timesteps.

4. OUTPUT GATE & HIDDEN STATE:
   o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)
   Filters the internal physical degradation memory to emit the external hidden vector h_t.
========================================================================================
"""

from pathlib import Path
from typing import Optional, Tuple
import json
import numpy as np

# Suppress verbose TensorFlow logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

from src.config import settings
from src.utils.logger import logger


class LSTMRULModel:
    """Deep Bidirectional LSTM model for 30-cycle multivariate time-series regression."""

    def __init__(
        self,
        sequence_length: int = settings.SEQUENCE_WINDOW_SIZE,
        num_features: int = len(settings.INFORMATIVE_SENSOR_COLS),
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model: Optional[tf.keras.Model] = None

    def build_model(self) -> tf.keras.Model:
        """Constructs Bidirectional LSTM deep architecture."""
        inputs = layers.Input(shape=(self.sequence_length, self.num_features), name="sensor_sequence_input")
        
        # Bidirectional LSTM Layer 1: processes both forward wear and backward context
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            name="bidirectional_lstm_1",
        )(inputs)
        x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_1")(x)

        # LSTM Layer 2: condenses sequential representations
        x = layers.LSTM(
            self.lstm_units // 2,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(1e-4),
            name="lstm_2",
        )(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_2")(x)

        # Dense projection head
        x = layers.Dense(32, activation="relu", name="dense_latent")(x)
        outputs = layers.Dense(1, activation="linear", name="rul_output")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="Sentinel_BiLSTM_RUL")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        self.model = model
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 35,
        batch_size: int = 64,
    ) -> tf.keras.callbacks.History:
        """Trains the LSTM network with early stopping and learning rate annealing."""
        if self.model is None:
            self.num_features = X_train.shape[2]
            self.sequence_length = X_train.shape[1]
            self.build_model()

        cb_list = [
            callbacks.EarlyStopping(
                monitor="val_loss" if validation_data is not None else "loss",
                patience=7,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if validation_data is not None else "loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        logger.info(
            f"Training BiLSTM on {X_train.shape[0]} sequences (shape: {X_train.shape}) for {epochs} epochs..."
        )
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=1,
        )
        logger.info("BiLSTM training completed.")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Runs inference on 3D sequence tensors."""
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")
        preds = self.model.predict(X, verbose=0).flatten()
        return np.maximum(preds, 0.0)

    def save(self, model_dir: Optional[Path] = None):
        """Saves Keras model and hyperparameter metadata."""
        model_dir = model_dir or settings.MODELS_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "lstm_model.keras"
        meta_path = model_dir / "lstm_metadata.json"

        self.model.save(model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "sequence_length": self.sequence_length,
                "num_features": self.num_features,
                "lstm_units": self.lstm_units,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
            }, f, indent=2)
        logger.info(f"Saved LSTM model to {model_path}")

    @classmethod
    def load(cls, model_dir: Optional[Path] = None) -> "LSTMRULModel":
        """Loads trained LSTM model and configuration."""
        model_dir = model_dir or settings.MODELS_DIR
        model_path = model_dir / "lstm_model.keras"
        meta_path = model_dir / "lstm_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No LSTM model found at {model_path}")

        meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)

        instance = cls(
            sequence_length=meta.get("sequence_length", settings.SEQUENCE_WINDOW_SIZE),
            num_features=meta.get("num_features", len(settings.INFORMATIVE_SENSOR_COLS)),
            lstm_units=meta.get("lstm_units", 64),
            dropout_rate=meta.get("dropout_rate", 0.2),
            learning_rate=meta.get("learning_rate", 0.001),
        )
        instance.model = tf.keras.models.load_model(model_path)
        logger.info(f"Successfully loaded LSTM model from {model_path}")
        return instance
