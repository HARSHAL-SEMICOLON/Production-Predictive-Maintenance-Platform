"""Live Factory Sensor Simulator for Sentinel AI.

Simulates live IoT edge sensors transmitting high-frequency multivariate telemetry
from turbofan engines to the central FastAPI gateway and Redis cache.
"""

import argparse
import json
import time
from typing import Generator, Dict, Any, Optional
import requests

from src.config import settings
from src.utils.logger import logger
from src.ingestion.cmapss_loader import CMAPSSDataLoader, C_MAPSS_NOMINAL_BASELINES


class TurbofanSensorSimulator:
    """Generates and streams realistic turbofan degradation telemetry."""

    def __init__(self, api_url: str = "http://127.0.0.1:8000/api/v1/telemetry"):
        self.api_url = api_url
        self.loader = CMAPSSDataLoader()

    def generate_engine_stream(
        self,
        engine_id: int = 1,
        start_cycle: int = 1,
        max_cycles: int = 180,
        inject_anomaly_at: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yields sequential sensor readings simulating an engine undergoing gradual wear."""
        # Try loading authentic test engine data if available
        test_file = settings.DATA_PROCESSED_DIR / "test_FD001.parquet"
        if test_file.exists():
            import pandas as pd
            df = pd.read_parquet(test_file)
            engine_df = df[df["engine_id"] == engine_id].sort_values(by="cycle")
            if len(engine_df) > 0:
                for _, row in engine_df.iterrows():
                    telemetry = row.to_dict()
                    # Clean out target columns if present
                    for k in ["RUL", "RUL_clipped", "is_critical"]:
                        telemetry.pop(k, None)
                    telemetry["engine_id"] = int(engine_id)
                    telemetry["cycle"] = int(telemetry["cycle"])
                    yield telemetry
                return

        # Synthetic degradation generator
        lifespan = max_cycles
        wear_knee = int(lifespan * 0.55)

        for cycle in range(start_cycle, lifespan + 1):
            wear_factor = 0.0
            if cycle > wear_knee:
                wear_factor = ((cycle - wear_knee) / (lifespan - wear_knee)) ** 1.8

            telemetry = {
                "engine_id": int(engine_id),
                "cycle": int(cycle),
                "setting_1": round(0.000 + float(np.random.normal(0, 0.001)), 4),
                "setting_2": round(0.000 + float(np.random.normal(0, 0.0002)), 4),
                "setting_3": 100.0,
            }

            for s_id in range(1, 22):
                s_name = f"sensor_{s_id}"
                mean_val, std_val = C_MAPSS_NOMINAL_BASELINES[s_name]
                drift_dir = 1.0 if s_id in [2, 3, 4, 8, 11, 13, 15, 17] else -1.0
                drift = (std_val * 4.5) * wear_factor * drift_dir if std_val > 0 else 0.0
                
                # Anomaly injection
                if inject_anomaly_at and cycle >= inject_anomaly_at:
                    if s_id in [2, 3, 4]:  # Severe thermal runaway
                        drift += std_val * 6.0
                    elif s_id in [8, 13]:  # Fan imbalance vibration
                        drift += std_val * 5.0

                noise = float(np.random.normal(0, std_val * 0.3)) if std_val > 0 else 0.0
                telemetry[s_name] = round(float(mean_val + drift + noise), 4)

            yield telemetry

    def stream_to_api(
        self,
        engine_id: int = 1,
        interval_seconds: float = 1.0,
        max_cycles: int = 100,
        inject_anomaly_at: Optional[int] = None,
    ):
        """Streams sensor readings to central API at regular intervals."""
        logger.info(
            f"Starting sensor simulation for Engine {engine_id} (interval: {interval_seconds}s, "
            f"anomaly at cycle: {inject_anomaly_at})..."
        )
        stream = self.generate_engine_stream(
            engine_id=engine_id,
            start_cycle=1,
            max_cycles=max_cycles,
            inject_anomaly_at=inject_anomaly_at,
        )

        for reading in stream:
            try:
                resp = requests.post(self.api_url, json=reading, timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    pred = data.get("prediction", {})
                    logger.info(
                        f"Engine {reading['engine_id']} | Cycle {reading['cycle']} | "
                        f"RUL: {pred.get('predicted_rul'):.1f} cycles | "
                        f"Risk: {pred.get('failure_probability', 0)*100:.1f}% | "
                        f"Status: {pred.get('health_status')}"
                    )
                else:
                    logger.warning(f"API returned status {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"Failed to transmit telemetry to {self.api_url}: {e}")

            time.sleep(interval_seconds)


if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser(description="Sentinel AI Sensor Simulator")
    parser.add_argument("--engine", type=int, default=1, help="Engine ID to simulate")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between readings")
    parser.add_argument("--max-cycles", type=int, default=120, help="Number of cycles to simulate")
    parser.add_argument("--anomaly-at", type=int, default=None, help="Inject anomaly at cycle number")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/api/v1/telemetry", help="Target API URL")
    args = parser.parse_args()

    sim = TurbofanSensorSimulator(api_url=args.url)
    sim.stream_to_api(
        engine_id=args.engine,
        interval_seconds=args.interval,
        max_cycles=args.max_cycles,
        inject_anomaly_at=args.anomaly_at,
    )
