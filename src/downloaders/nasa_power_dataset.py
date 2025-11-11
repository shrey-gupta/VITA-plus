from huggingface_hub import snapshot_download
import os
import argparse
from src.utils.constants import DEFAULT_DATA_DIR
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    os.makedirs(args.data_dir + "nasa_power", exist_ok=True)

    logger.info(f"Downloading NASA Power dataset to {args.data_dir + 'nasa_power'}...")
    snapshot_download(
        repo_id="notadib/NASA-Power-Daily-Weather",
        repo_type="dataset",
        allow_patterns="pytorch/*weekly*",
        local_dir=args.data_dir + "nasa_power",
    )
    os.rename(
        args.data_dir + "nasa_power/pytorch", args.data_dir + "nasa_power/processed"
    )
    logger.info(f"NASA Power dataset downloaded to {args.data_dir + 'nasa_power'}")
