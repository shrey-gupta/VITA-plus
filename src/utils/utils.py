import torch.optim as optim
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
import logging
from argparse import ArgumentParser
import math


def get_scheduler(optimizer, num_warmup_epochs, total_epochs, decay_factor=None):
    """
    Create a learning rate scheduler with warmup followed by cosine or exponential annealing.

    Args:
        optimizer: PyTorch optimizer
        num_warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        decay_factor: If None, use cosine annealing. If provided, use exponential annealing with this decay factor.

    Returns:
        PyTorch LR scheduler
    """

    def _cosine_annealing_lr(num_warmup_epochs, total_epochs):
        def lr_function(current_epoch):
            if current_epoch < num_warmup_epochs:
                # Linear warmup
                return float(current_epoch) / float(max(1, num_warmup_epochs))
            else:
                # Cosine annealing after warmup
                progress = (current_epoch - num_warmup_epochs) / (
                    total_epochs - num_warmup_epochs
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_function

    def _exponential_annealing_lr(num_warmup_epochs, total_epochs, decay_factor):
        def lr_function(current_epoch):
            if current_epoch < num_warmup_epochs:
                # Linear warmup
                return float(current_epoch) / float(max(1, num_warmup_epochs))
            else:
                # Exponential annealing after warmup
                epochs_after_warmup = current_epoch - num_warmup_epochs
                return decay_factor**epochs_after_warmup

        return lr_function

    if decay_factor is None:
        # Use cosine annealing
        lr_lambda = _cosine_annealing_lr(num_warmup_epochs, total_epochs)
    else:
        # Use exponential annealing
        lr_lambda = _exponential_annealing_lr(
            num_warmup_epochs, total_epochs, decay_factor
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def normalize_year_interval_coords(year, interval, coords):
    """
    Normalize year, interval, and coordinates to be used in the model.
    """
    year = (year - 1970) / 100.0
    interval = interval / 30.0
    # Create a copy to avoid in-place modification
    coords = coords.clone()
    # training dataset covers ~2x more lat than lng these stds bring them to same range
    coords[:, 0] = coords[:, 0] / 360
    coords[:, 1] = (coords[:, 1]) / 180
    return year, interval, coords


def setup_distributed():
    """Initialize distributed training environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Initialize the process group
        dist.init_process_group(backend="nccl")

        # Set device for this process
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU training
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


# Configure logging only for rank 0
def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def get_model_params(model_size: str):
    if model_size == "mini":
        model_size_params = {"num_heads": 4, "num_layers": 2, "hidden_dim_factor": 12}
    elif model_size == "small":
        model_size_params = {"num_heads": 10, "num_layers": 4, "hidden_dim_factor": 20}
    elif model_size == "medium":
        model_size_params = {"num_heads": 12, "num_layers": 6, "hidden_dim_factor": 28}
    elif model_size == "large":
        model_size_params = {"num_heads": 16, "num_layers": 8, "hidden_dim_factor": 36}
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    return model_size_params


def parse_args(parser: ArgumentParser) -> dict:
    args = parser.parse_args()
    args_dict = vars(args)

    logger = logging.getLogger(__name__)
    logger.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logger.info(f"{arg}: {value}")

    # Model size configuration
    model_size = args.model_size.lower()
    model_size_params = get_model_params(model_size)

    args_dict["model_size_params"] = model_size_params

    return args_dict
