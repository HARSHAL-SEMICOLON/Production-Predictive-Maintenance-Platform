"""Script to download or generate NASA C-MAPSS dataset."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.cmapss_loader import CMAPSSDataLoader
from src.utils.logger import logger


def main():
    logger.info("Initializing NASA C-MAPSS FD001 dataset preparation...")
    loader = CMAPSSDataLoader()
    train_df, test_df, rul_df = loader.load_or_generate_data("FD001")
    logger.info(
        f"Dataset Ready: {len(train_df)} training cycles, {len(test_df)} test cycles, "
        f"{len(rul_df)} ground-truth test RUL labels."
    )


if __name__ == "__main__":
    main()
