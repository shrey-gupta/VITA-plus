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

    os.makedirs(args.data_dir + "khaki_corn_belt", exist_ok=True)

    logger.info(
        f"Downloading Khaki Corn Belt dataset to {args.data_dir + 'khaki_corn_belt'}..."
    )
    snapshot_download(
        repo_id="notadib/usa-corn-belt-crop-yield",
        repo_type="dataset",
        allow_patterns="khaki_multi_crop_yield.csv",
        local_dir=args.data_dir + "khaki_corn_belt",
    )
    logger.info(
        f"Khaki Corn Belt dataset downloaded to {args.data_dir + 'khaki_corn_belt'}"
    )
