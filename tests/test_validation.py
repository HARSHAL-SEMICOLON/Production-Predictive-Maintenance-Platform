"""Unit tests for Sentinel AI data validation schemas and checks."""

import pytest
import pandas as pd
from pydantic import ValidationError

from src.validation.schemas import SensorTelemetryIn, PredictionRequest
from src.validation.data_validator import DataValidator


def test_sensor_telemetry_valid():
    payload = {
        "engine_id": 1,
        "cycle": 10,
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
    telemetry = SensorTelemetryIn(**payload)
    assert telemetry.engine_id == 1
    assert telemetry.cycle == 10
    assert telemetry.sensor_3 == 1588.0


def test_sensor_telemetry_invalid_negative_engine_id():
    with pytest.raises(ValidationError):
        SensorTelemetryIn(engine_id=-5, cycle=1)


def test_sensor_telemetry_out_of_bounds_sensor():
    with pytest.raises(ValidationError):
        # Sensor 2 (LPC temp) physical bound is [400, 800]
        SensorTelemetryIn(engine_id=1, cycle=1, sensor_2=99999.0)


def test_data_validator_rule_checks():
    # Valid record
    valid_data = {"engine_id": 1, "cycle": 5, "sensor_3": 1588.0, "sensor_8": 2388.0}
    is_valid, errors = DataValidator.validate_telemetry_dict(valid_data)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid record (out of bounds)
    invalid_data = {"engine_id": 1, "cycle": 5, "sensor_3": 9999.0}
    is_valid, errors = DataValidator.validate_telemetry_dict(invalid_data)
    assert is_valid is False
    assert len(errors) > 0


def test_batch_dataframe_validation():
    df_valid = pd.DataFrame([
        {"engine_id": 1, "cycle": 1, "sensor_2": 642.5, "sensor_3": 1588.0},
        {"engine_id": 1, "cycle": 2, "sensor_2": 643.0, "sensor_3": 1589.0},
    ])
    report = DataValidator.validate_batch_dataframe(df_valid)
    assert report["is_valid"] is True
    assert report["missing_values_count"] == 0
    assert report["duplicate_cycles_count"] == 0
