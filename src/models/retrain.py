"""Automated Retraining Module for Sentinel AI.

Triggered when Evidently AI detects significant data or feature drift,
or via manual API trigger from operations team.
"""

from typing import Dict, Any
from src.utils.logger import logger
from src.models.train import train_models


def trigger_automated_retraining(reason: str = "Drift detection alert") -> Dict[str, Any]:
    """Orchestrates model retraining, evaluates against production criteria, and updates registry."""
    logger.info(f"=== Automated Retraining Triggered. Reason: {reason} ===")
    
    # Run full training pipeline
    production_meta = train_models(dataset_name="FD001", lstm_epochs=20, quick_run=False)
    
    logger.info(f"Retraining successful. New champion: {production_meta['champion_model']}")
    return {
        "status": "success",
        "reason": reason,
        "new_champion": production_meta["champion_model"],
        "metrics": production_meta[f"{production_meta['champion_model']}_metrics"],
        "timestamp": production_meta["trained_at"],
    }


if __name__ == "__main__":
    trigger_automated_retraining("CLI test trigger")
