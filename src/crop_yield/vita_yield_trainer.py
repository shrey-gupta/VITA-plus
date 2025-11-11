import torch
import torch.nn as nn
import pandas as pd
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader
from src.base.base_trainer import BaseTrainer
from src.utils.constants import (
    DEFAULT_DATA_DIR,
    TOTAL_WEATHER_VARS,
    EXTREME_YEARS,
    TEST_YEARS,
)
from src.crop_yield.vita_yield_model import VITAYieldModel
from src.dataloaders.khaki_corn_belt_dataloader import (
    get_train_test_loaders,
    read_khaki_corn_belt_dataset,
)
from src.base.cross_validator import CrossValidator
from src.utils.losses import (
    compute_gaussian_kl_divergence,
)
import os

FOLD_IDX = 0


def _reset_fold_index():
    """Reset the global fold index for a new cross-validation run."""
    global FOLD_IDX
    FOLD_IDX = 0


class VITAYieldTrainer(BaseTrainer):
    """
    Trainer class for VITA-based crop yield prediction models.

    PUBLIC API METHODS (for users):
        - train(): Inherited from BaseTrainer - main training entry point
        - save_checkpoint(): Inherited from BaseTrainer
        - load_checkpoint(): Inherited from BaseTrainer

    ABSTRACT METHOD IMPLEMENTATIONS (required by BaseTrainer):
        - get_dataloaders(): Get train/validation data loaders
        - compute_train_loss(): Compute training loss for a batch
        - compute_validation_loss(): Compute validation loss for a batch
    """

    def __init__(
        self,
        crop_df: pd.DataFrame,
        n_past_years: int,
        n_train_years: int,
        beta: float,
        crop_type: str,
        test_year: Optional[int] = None,
        test_type: str = "extreme",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.crop_df = crop_df
        self.n_past_years = n_past_years
        self.n_train_years = n_train_years
        self.beta = beta
        self.crop_type = crop_type
        self.test_type = test_type
        self.output_json["model_config"]["beta"] = beta

        self.criterion = nn.MSELoss(reduction="mean")

        # Select test years based on test_type
        if test_type == "extreme":
            test_years = EXTREME_YEARS.get("usa", {}).get(crop_type)
            if test_years is None:
                raise ValueError(f"No extreme years found for {crop_type}.")
        elif test_type == "overall":
            test_years = TEST_YEARS
        elif test_type == "ahead_pred":
            test_years = TEST_YEARS
        else:
            raise ValueError(
                f"Unknown test_type: {test_type}. Choose 'extreme', 'overall', or 'ahead_pred'."
            )

        self.logger.info(f"Test type: {test_type} on years {test_years}")

        if self.rank == 0:
            self.model_dir = DEFAULT_DATA_DIR + "trained_models/crop_yield/"
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

            # Override loss collection to match VITA's expected keys
            self.output_json["losses"] = {
                "train": {
                    "total_loss": [],
                    "yield": [],
                    "reconstruction": [],
                    "kl_term": [],
                },
                "val": {
                    "total_loss": [],
                },
            }

        if test_year is not None:
            self.test_year = test_year
            self.logger.info(
                f"Single test year mode - Testing on year: {self.test_year}"
            )
        else:
            global FOLD_IDX
            if FOLD_IDX >= len(test_years):
                raise ValueError(
                    f"FOLD_IDX ({FOLD_IDX}) exceeds TEST_YEARS length ({len(test_years)}). Call _reset_fold_index() before starting new cross-validation."
                )
            self.test_year = test_years[FOLD_IDX]
            FOLD_IDX += 1
            self.logger.info(
                f"Cross-validation mode ({test_type}) - Testing on year: {self.test_year}"
            )

        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def get_dataloaders(self, shuffle: bool = False) -> Tuple[DataLoader, DataLoader]:  # type: ignore
        """Get data loaders for training/validation."""
        if self.train_loader is not None and self.test_loader is not None:
            return self.train_loader, self.test_loader

        test_gap = 4 if self.test_type == "ahead_pred" else 0
        train_loader, test_loader = get_train_test_loaders(
            self.crop_df,
            self.n_train_years,
            self.test_year,
            self.n_past_years,
            self.batch_size,
            shuffle,
            num_workers=0 if self.world_size > 1 else 8,
            crop_type=self.crop_type,
            test_gap=test_gap,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        return train_loader, test_loader

    def compute_kl_loss(
        self,
        weather_feature_mask: torch.Tensor,
        z: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss between posterior and sinusoidal prior."""
        kl_term = compute_gaussian_kl_divergence(
            feature_mask=weather_feature_mask,
            mu_x=mu_x,
            var_x=var_x,
            mu_p=mu_p,
            var_p=var_p,
        )
        return kl_term

    def compute_elbo_loss(
        self,
        weather: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        target_yield: torch.Tensor,
        yield_pred: torch.Tensor,
        z: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
        log_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the ELBO loss for VITA yield prediction.

        Args:
            weather: Original weather data
            weather_feature_mask: Boolean mask for weather features
            target_yield: Ground truth yield values
            yield_pred: Predicted yield values
            z: Sampled weather representations
            mu_x: Mean of weather representations
            var_x: Variance of weather representations
            mu_p: Mean of sinusoidal prior
            var_p: Variance of sinusoidal prior
        """
        # 1. Yield term: MSE between predicted and target yield
        yield_loss = self.criterion(yield_pred.squeeze(), target_yield.squeeze())

        beta = self._current_beta()

        # 2. Reconstruction term: Gaussian negative log-likelihood for weather features
        reconstruction_term = torch.tensor(0.0)

        # 3. KL divergence term: � * KL(q(z|x) || p(z))
        kl_term = (
            beta
            * self.compute_kl_loss(
                weather_feature_mask, z, mu_x, var_x, mu_p, var_p
            ).mean()
        )

        if log_losses:
            self.logger.info(f"Yield Loss: {yield_loss.item():.6f}")
            self.logger.info(f"Reconstruction Term: {reconstruction_term.item():.6f}")
            self.logger.info(f"KL Term: {kl_term.item():.6f}")

        total_loss = yield_loss + reconstruction_term + kl_term

        return {
            "total_loss": total_loss,
            "yield": yield_loss,
            "reconstruction": reconstruction_term,
            "kl_term": kl_term,
        }

    def compute_train_loss(  # type: ignore
        self,
        padded_weather,
        coord_processed,
        year_expanded,
        interval,
        weather_feature_mask,
        practices,
        soil,
        y_past,
        target_yield,
    ) -> Dict[str, torch.Tensor]:
        """Compute variational training loss for VITA yield prediction."""
        # Forward pass through VITA model
        # Returns (yield_pred, z, mu_x, var_x, mu_p, var_p)
        model_outputs = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
        )

        return self.compute_elbo_loss(
            padded_weather, weather_feature_mask, target_yield, *model_outputs
        )

    def compute_validation_loss(  # type: ignore
        self,
        padded_weather,
        coord_processed,
        year_expanded,
        interval,
        weather_feature_mask,
        practices,
        soil,
        y_past,
        target_yield,
    ) -> Dict[str, torch.Tensor]:
        """Compute variational validation loss for VITA yield prediction."""
        with torch.no_grad():
            model_outputs = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                y_past,
            )

        components = self.compute_elbo_loss(
            padded_weather, weather_feature_mask, target_yield, *model_outputs
        )

        # Return RMSE for validation
        return {"total_loss": components["yield"] ** 0.5}

    def _current_beta(self):
        return self.beta


# =============================================================================
# PUBLIC API FUNCTIONS (for users)
# =============================================================================


def _create_yield_training_setup(args_dict):
    """
    Helper function to create common training setup for all yield trainers.
    Returns common parameters needed by all yield training loops.
    """
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    if torch.backends.mps.is_available():
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Detected MPS device, forcing CPU for DGL compatibility")
    else:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args_dict.get("data_dir", DEFAULT_DATA_DIR)
    crop_df = read_khaki_corn_belt_dataset(data_dir)

    if args_dict.get("test_year") is not None:
        cross_validation_k = 1
    else:
        cross_validation_k = len(TEST_YEARS)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "crop_df": crop_df,
        "cross_validation_k": cross_validation_k,
        "beta": args_dict["beta"],
        "test_year": args_dict.get("test_year"),
        "test_type": args_dict.get("test_type", "extreme"),
    }


def _run_yield_cross_validation(
    setup_params,
    model_class,
    trainer_class,
    model_name,
    args_dict,
    extra_trainer_kwargs=None,
    extra_model_kwargs=None,
):
    """Helper function to run cross-validation for yield prediction models."""
    if setup_params["test_year"] is None:
        _reset_fold_index()

    model_kwargs = {
        "name": model_name,
        "device": setup_params["device"],
        "weather_dim": TOTAL_WEATHER_VARS,
        "n_past_years": args_dict["n_past_years"],
        **args_dict["model_size_params"],
    }

    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)

    trainer_kwargs = {
        "crop_df": setup_params["crop_df"],
        "n_past_years": args_dict["n_past_years"],
        "n_train_years": args_dict["n_train_years"],
        "beta": args_dict["beta"],
        "crop_type": args_dict["crop_type"],
        "test_year": setup_params["test_year"],
        "test_type": setup_params["test_type"],
        "data_dir": args_dict["data_dir"],
        "batch_size": args_dict["batch_size"],
        "num_epochs": args_dict["n_epochs"],
        "init_lr": args_dict["init_lr"],
        "num_warmup_epochs": args_dict["n_warmup_epochs"],
        "decay_factor": args_dict["decay_factor"],
        "pretrained_model_path": args_dict["pretrained_model_path"],
        "resume_from_checkpoint": args_dict["resume_from_checkpoint"],
        "rank": setup_params["rank"],
        "world_size": setup_params["world_size"],
        "local_rank": setup_params["local_rank"],
    }

    if extra_trainer_kwargs:
        trainer_kwargs.update(extra_trainer_kwargs)

    cross_validator = CrossValidator(
        model_class=model_class,
        model_kwargs=model_kwargs,
        trainer_class=trainer_class,
        trainer_kwargs=trainer_kwargs,
        k_folds=setup_params["cross_validation_k"],
    )

    return cross_validator.run_cross_validation()


def vita_yield_training_loop(args_dict):
    """
    VITA training loop using the VITAYieldTrainer class.
    Initializes the model internally and handles all training.

    Args:
        args_dict: Arguments dictionary
    """
    setup_params = _create_yield_training_setup(args_dict)

    extra_model_kwargs = {"k": args_dict["k"]}
    extra_trainer_kwargs = {"beta": args_dict["beta"]}

    return _run_yield_cross_validation(
        setup_params=setup_params,
        model_class=VITAYieldModel,
        trainer_class=VITAYieldTrainer,
        model_name=f"vita_{args_dict['crop_type']}_yield",
        args_dict=args_dict,
        extra_trainer_kwargs=extra_trainer_kwargs,
        extra_model_kwargs=extra_model_kwargs,
    )
