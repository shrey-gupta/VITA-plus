import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from src.pretraining.vita_trainer import vita_training_loop
from src.utils.utils import setup_distributed, cleanup_distributed, setup_logging
from src.utils.utils import parse_args
from src.utils.constants import DEFAULT_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    help="path to data directory",
    default=DEFAULT_DATA_DIR,
    type=str,
)
parser.add_argument(
    "--resume-from-checkpoint",
    help="path to resume from checkpoint",
    default=None,
    type=str,
)
parser.add_argument(
    "--pretrained-model-path",
    help="path to pretrained model to load before training",
    default=None,
    type=str,
)
parser.add_argument("--batch-size", help="batch size", default=256, type=int)
parser.add_argument(
    "--n-masked-features",
    help="number of masked features, the rest of the features are input features",
    default=10,
    type=int,
)
parser.add_argument(
    "--n-epochs", help="number of training epochs", default=100, type=int
)
parser.add_argument(
    "--init-lr", help="initial learning rate", default=0.0005, type=float
)
parser.add_argument(
    "--n-warmup-epochs", help="number of warm-up epochs", default=10, type=float
)
parser.add_argument(
    "--decay-factor",
    help="exponential learning rate decay factor after warmup",
    default=0.99,
    type=float,
)
parser.add_argument(
    "--model-size",
    help="model size mini (60k), small (2M), medium (8M), and large (56M)",
    default="small",
    type=str,
)
parser.add_argument(
    "--alpha",
    help="parameter for sinusoidal prior loss weighting",
    default=0.5,
    type=float,
)
parser.add_argument(
    "--seed",
    help="random seed for reproducibility",
    default=1234,
    type=int,
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Setup logging
    setup_logging(rank)

    try:
        args_dict = parse_args(parser)

        # Set random seed for reproducibility
        set_seed(args_dict["seed"])

        # Add distributed training info to args
        args_dict["rank"] = rank
        args_dict["world_size"] = world_size
        args_dict["local_rank"] = local_rank
        vita_training_loop(args_dict)
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()
