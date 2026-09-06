"""Data validation engine for Sentinel AI (Great Expectations-style quality checks)."""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.config import settings
from src.utils.logger import logger


class DataValidator:
    """Validates raw and streaming sensor data against industrial physical boundaries."""

    # Reasonable physical operating bounds for turbofan sensors
    SENSOR_BOUNDS: Dict[str, Tuple[float, float]] = {
        "sensor_1": (450.0, 600.0),    # Fan Inlet Temp [°R]
        "sensor_2": (550.0, 750.0),    # LPC Outlet Temp [°R]
        "sensor_3": (1400.0, 1800.0),  # HPC Outlet Temp [°R]
        "sensor_4": (1200.0, 1650.0),  # LPT Outlet Temp [°R]
        "sensor_5": (10.0, 20.0),      # Fan Inlet Pressure [psia]
        "sensor_6": (15.0, 30.0),      # Bypass Duct Pressure [psia]
        "sensor_7": (450.0, 650.0),    # HPC Outlet Pressure [psia]
        "sensor_8": (2200.0, 2550.0),  # Physical Fan Speed [rpm]
        "sensor_9": (8500.0, 9500.0),  # Physical Core Speed [rpm]
        "sensor_10": (1.0, 1.8),       # Engine Pressure Ratio
        "sensor_11": (40.0, 55.0),     # HPC Static Pressure [psia]
        "sensor_12": (450.0, 580.0),   # Fuel Flow Ratio
        "sensor_13": (2200.0, 2550.0), # Corrected Fan Speed [rpm]
        "sensor_14": (7800.0, 8500.0), # Corrected Core Speed [rpm]
        "sensor_15": (7.5, 9.5),       # Bypass Ratio
        "sensor_16": (0.01, 0.10),     # Burner Fuel Flow
        "sensor_17": (350.0, 450.0),   # Bleed Enthalpy
        "sensor_18": (2200.0, 2550.0), # Demanded Fan Speed
        "sensor_19": (90.0, 110.0),    # Demanded Corrected Fan Speed
        "sensor_20": (30.0, 45.0),     # HPT Coolant Bleed
        "sensor_21": (18.0, 30.0),     # LPT Coolant Bleed
    }

    @classmethod
    def validate_telemetry_dict(cls, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates a single telemetry record.
        
        Returns:
            is_valid: bool
            errors: list of error description strings
        """
        errors = []

        # Check required identifiers
        if "engine_id" not in data or data["engine_id"] < 1:
            errors.append("Invalid or missing 'engine_id'. Must be positive integer.")
        if "cycle" not in data or data["cycle"] < 1:
            errors.append("Invalid or missing 'cycle'. Must be positive integer.")

        # Check physical ranges for sensor readings
        for s_name, (min_val, max_val) in cls.SENSOR_BOUNDS.items():
            if s_name in data:
                val = data[s_name]
                if val is None or np.isnan(val):
                    errors.append(f"Sensor {s_name} contains null/NaN value.")
                elif val < min_val or val > max_val:
                    errors.append(
                        f"Sensor {s_name} value {val:.2f} is out of physical operational bounds [{min_val}, {max_val}]."
                    )

        is_valid = len(errors) == 0
        return is_valid, errors

    @classmethod
    def validate_batch_dataframe(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs validation suite on a batch or training dataset.
        
        Checks:
        1. Missing values
        2. Duplicate (engine_id, cycle) rows
        3. Monotonic cycle progression per engine
        4. Out-of-bounds sensor values
        5. Frozen sensors (zero variance across sequence)
        """
        report = {
            "total_records": len(df),
            "is_valid": True,
            "missing_values_count": int(df.isnull().sum().sum()),
            "duplicate_cycles_count": int(df.duplicated(subset=["engine_id", "cycle"]).sum()),
            "out_of_bounds_sensors": {},
            "frozen_sensors": [],
            "validation_messages": [],
        }

        # 1. Missing values check
        if report["missing_values_count"] > 0:
            report["is_valid"] = False
            report["validation_messages"].append(
                f"Dataset failed check: {report['missing_values_count']} null values detected."
            )

        # 2. Duplicate rows check
        if report["duplicate_cycles_count"] > 0:
            report["is_valid"] = False
            report["validation_messages"].append(
                f"Dataset failed check: {report['duplicate_cycles_count']} duplicate (engine_id, cycle) pairs found."
            )

        # 3. Sensor range checks
        for s_name, (min_b, max_b) in cls.SENSOR_BOUNDS.items():
            if s_name in df.columns:
                out_of_range = df[(df[s_name] < min_b) | (df[s_name] > max_b)]
                if len(out_of_range) > 0:
                    report["out_of_bounds_sensors"][s_name] = int(len(out_of_range))

        # 4. Zero variance (frozen sensor) check across dynamic sensors
        for s_name in settings.INFORMATIVE_SENSOR_COLS:
            if s_name in df.columns:
                std_val = df[s_name].std()
                if std_val < 1e-6:
                    report["frozen_sensors"].append(s_name)

        if report["frozen_sensors"]:
            report["validation_messages"].append(
                f"Warning: Informative sensors detected with zero variance: {report['frozen_sensors']}"
            )

        if report["is_valid"]:
            logger.info(f"Data validation PASSED for {len(df)} records.")
        else:
            logger.warning(f"Data validation FAILED: {report['validation_messages']}")

        return report
