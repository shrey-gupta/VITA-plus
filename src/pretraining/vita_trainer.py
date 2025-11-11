import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.base.base_trainer import BaseTrainer
from src.base.vita import VITA
from src.dataloaders.nasa_power_dataloader import nasa_power_dataloader
from src.utils.constants import TOTAL_WEATHER_VARS
from src.utils.losses import compute_gaussian_kl_divergence, gaussian_log_likelihood


class VITATrainer(BaseTrainer):
    """
    VITA pretrainer for weather data with VAE-style loss.
    """

    def __init__(
        self,
        model: VITA,
        n_masked_features: int,
        alpha: float,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.n_masked_features = n_masked_features
        self.alpha = alpha
        self.output_json["model_config"]["n_masked_features"] = n_masked_features
        self.output_json["model_config"]["alpha"] = alpha
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
            "val": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
        }

    def compute_kl_loss(
        self,
        weather: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss between posterior and sinusoidal prior."""
        kl_term = compute_gaussian_kl_divergence(
            weather_feature_mask, mu_x, var_x, mu_p, var_p
        )
        return kl_term

    def compute_elbo_loss(
        self,
        weather: torch.Tensor,
        feature_mask: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss with reconstruction and KL terms."""
        # Truncate weather and feature_mask to match model output length
        seq_len = mu_x.shape[1]
        weather = weather[:, :seq_len, :]
        feature_mask = feature_mask[:, :seq_len, :]

        n_masked_features = feature_mask.sum(dim=(1, 2)).float().mean()
        reconstruction_term = (
            -gaussian_log_likelihood(weather, mu_x, var_x, feature_mask)
            / n_masked_features
        ).mean()
        kl_term = (
            self.alpha
            * self.compute_kl_loss(
                weather, feature_mask, mu_x, var_x, mu_p, var_p
            ).mean()
        ) / n_masked_features

        total_loss = reconstruction_term + kl_term

        return {
            "total_loss": total_loss,
            "reconstruction": reconstruction_term,
            "kl_term": kl_term,
        }

    def compute_train_loss(  # type: ignore
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute VITA training loss using VAE-style loss function."""
        model_outputs = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )
        loss_dict = self.compute_elbo_loss(weather, feature_mask, *model_outputs)
        return loss_dict

    def compute_validation_loss(  # type: ignore
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute VITA validation loss using VAE-style loss function."""
        model_outputs = self.model(
            weather, coords, year, interval, weather_feature_mask=feature_mask
        )
        loss_dict = self.compute_elbo_loss(weather, feature_mask, *model_outputs)
        return loss_dict

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:  # type: ignore
        """Get data loaders for training/validation."""

        current_n_masked = self._get_n_masked_features(
            self.current_epoch, self.n_masked_features
        )

        train_loader = nasa_power_dataloader(
            self.batch_size,
            self.data_dir,
            split="train",
            shuffle=shuffle,
            n_masked_features=current_n_masked,
            world_size=self.world_size,
            rank=self.rank,
        )

        val_loader = nasa_power_dataloader(
            self.batch_size,
            self.data_dir,
            split="validation",
            shuffle=False,
            n_masked_features=current_n_masked,
            world_size=self.world_size,
            rank=self.rank,
        )

        return train_loader, val_loader


def vita_training_loop(args_dict):
    """
    VITA pretraining loop using the VITATrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize vita model
    model = VITA(
        weather_dim=TOTAL_WEATHER_VARS,
        output_dim=TOTAL_WEATHER_VARS,
        device=device,
        **args_dict["model_size_params"],
    ).to(device)

    if rank == 0:
        logging.info(str(model))

    trainer = VITATrainer(
        model=model,
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        data_dir=args_dict["data_dir"],
        init_lr=args_dict["init_lr"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        pretrained_model_path=args_dict["pretrained_model_path"],
        n_masked_features=args_dict["n_masked_features"],
        alpha=args_dict["alpha"],
        resume_from_checkpoint=args_dict.get("resume_from_checkpoint"),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    return trainer.train()
