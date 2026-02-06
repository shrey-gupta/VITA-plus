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
    GRIDMET_TO_NASA_IDX,
)
from src.base.cross_validator import CrossValidator
from src.utils.losses import (
    compute_gaussian_kl_divergence,
)
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


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
        year_weights: Optional[Dict[int, float]] = None,
        cvar_frac: float = 0.0,
        feature_dropout_prob: float = 0.0,
        feature_dropout_protect_indices: Optional[list[int]] = None,
        drift_weight_strength: float = 0.0,
        drift_min_weight: float = 0.2,
        drift_features: Optional[list[str]] = None,
        drift_target_year: Optional[int] = None,
        attn_bias_strength: float = 0.0,
        target_stats_start_week: int = 15,
        target_stats_max_week: int = 35,
        weather_vars: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.crop_df = crop_df
        self.n_past_years = n_past_years
        self.n_train_years = n_train_years
        self.beta = beta
        self.crop_type = crop_type
        self.test_type = test_type
        self.year_weights = year_weights or {}
        self.cvar_frac = cvar_frac
        self.feature_dropout_prob = feature_dropout_prob
        self.feature_dropout_protect_indices = (
            feature_dropout_protect_indices if feature_dropout_protect_indices is not None else []
        )
        self.drift_weight_strength = drift_weight_strength
        self.drift_min_weight = drift_min_weight
        self.drift_features = drift_features or ["pr", "vpd"]
        self.drift_target_year = drift_target_year if drift_target_year is not None else test_year
        self.attn_bias_strength = attn_bias_strength
        self.target_stats_start_week = target_stats_start_week
        self.target_stats_max_week = target_stats_max_week
        self.weather_vars = weather_vars

        # Drift reference stats (set later)
        self._target_feature_means: Optional[torch.Tensor] = None
        self._target_feature_stds: Optional[torch.Tensor] = None
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

        # Precompute drift reference stats if enabled
        if self.drift_weight_strength > 0.0 or self.attn_bias_strength > 0.0:
            try:
                self._compute_target_feature_stats()
            except Exception as exc:
                self.logger.warning(f"Failed to compute drift target stats; disabling drift weighting. Error: {exc}")
                self.drift_weight_strength = 0.0

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
            num_workers=1,
            crop_type=self.crop_type,
            test_gap=test_gap,
            weather_vars=self.weather_vars,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        return train_loader, test_loader

    def _apply_feature_dropout(self, padded_weather: torch.Tensor) -> torch.Tensor:
        """Stochastically drop non-protected weather features during training."""
        if self.feature_dropout_prob <= 0.0:
            return padded_weather

        device = padded_weather.device
        feature_keep = torch.ones((TOTAL_WEATHER_VARS,), device=device)
        if self.feature_dropout_protect_indices:
            feature_keep[self.feature_dropout_protect_indices] = 1.0

        # Sample keep/drop mask per feature (shared across time steps)
        drop_mask = torch.bernoulli(
            torch.full((TOTAL_WEATHER_VARS,), 1.0 - self.feature_dropout_prob, device=device)
        )
        # Always keep protected features
        drop_mask[self.feature_dropout_protect_indices] = 1.0

        # Broadcast to all time steps
        drop_mask = drop_mask.unsqueeze(0)
        return padded_weather * drop_mask

    def _compute_target_feature_stats(self):
        """Compute target year feature means/stds for drift weighting."""
        if self.drift_target_year is None:
            return
        df_target = self.crop_df[self.crop_df["year"] == self.drift_target_year]
        if df_target.empty:
            raise ValueError(f"No data found for target year {self.drift_target_year} to compute drift stats.")

        means = []
        stds = []
        # Clamp window to valid ordering and available columns
        start_week = max(1, int(self.target_stats_start_week))
        end_week = max(start_week, int(self.target_stats_max_week))

        for feat in self.drift_features:
            cols = [
                f"{feat}_week_{j}"
                for j in range(start_week, end_week + 1)
                if f"{feat}_week_{j}" in df_target.columns
            ]
            if not cols:
                raise ValueError(f"No columns found for feature '{feat}' to compute drift stats.")
            values = df_target[cols].values.astype("float32").reshape(-1)
            means.append(float(values.mean()))
            stds.append(float(values.std() + 1e-6))

        self._target_feature_means = torch.tensor(means, dtype=torch.float32, device=self.device)
        self._target_feature_stds = torch.tensor(stds, dtype=torch.float32, device=self.device)
        self.logger.info(
            f"Drift reference (year {self.drift_target_year}, weeks {start_week}-{end_week}) for features {self.drift_features}: "
            f"means={self._target_feature_means.cpu().numpy()}, stds={self._target_feature_stds.cpu().numpy()}"
        )

    def _compute_sample_feature_means(
        self, padded_weather: torch.Tensor, weather_feature_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample means for selected features, masking padded steps."""
        if not self.drift_features:
            return torch.zeros((padded_weather.size(0), 0), device=padded_weather.device)

        indices = torch.tensor(
            [GRIDMET_TO_NASA_IDX[f] for f in self.drift_features],
            device=padded_weather.device,
            dtype=torch.long,
        )
        # padded_weather: [batch, steps, feats]
        selected = padded_weather[:, :, indices]  # [B, T, F]
        valid = (~weather_feature_mask[:, :, indices]).float()  # mask False where valid
        numer = (selected * valid).sum(dim=1)  # [B, F]
        denom = valid.sum(dim=1).clamp(min=1.0)
        return numer / denom

    def _batch_drift_weights(
        self, padded_weather: torch.Tensor, weather_feature_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Compute importance weights based on distance to target feature stats."""
        if self.drift_weight_strength <= 0.0 or self._target_feature_means is None:
            return None

        feature_means = self._compute_sample_feature_means(padded_weather, weather_feature_mask)
        # z-distance to target
        z = torch.abs(feature_means - self._target_feature_means) / self._target_feature_stds
        dist = z.mean(dim=1)  # average across selected features
        weights = torch.exp(-self.drift_weight_strength * dist)
        return torch.clamp(weights, min=self.drift_min_weight)

    def _token_attention_bias(
        self, padded_weather: torch.Tensor, weather_feature_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Compute per-token bias to favor timesteps whose selected features match target profile."""
        if self.attn_bias_strength <= 0.0:
            return None
        if self._target_feature_means is None or self._target_feature_stds is None:
            return None

        device = padded_weather.device
        indices = torch.tensor(
            [GRIDMET_TO_NASA_IDX[f] for f in self.drift_features],
            device=device,
            dtype=torch.long,
        )
        feats = padded_weather[:, :, indices]  # [B, T, F]
        mask = (~weather_feature_mask[:, :, indices]).float()  # 1 where valid
        # z-distance per feature per token
        z = torch.abs((feats - self._target_feature_means) / self._target_feature_stds)
        # average over selected features (ignore masked)
        denom = mask.sum(dim=2).clamp(min=1.0)
        dist = (z * mask).sum(dim=2) / denom  # [B, T]
        bias = -self.attn_bias_strength * dist

        # Zero out bias outside the configured week window (e.g., beyond week 35)
        seq_len = padded_weather.size(1)
        week_idx = torch.arange(seq_len, device=device) % 52 + 1  # 1-based weeks
        window_mask = (week_idx >= self.target_stats_start_week) & (
            week_idx <= self.target_stats_max_week
        )
        bias = bias * window_mask.unsqueeze(0)
        return bias.unsqueeze(-1)  # [B, T, 1]

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
        sample_weights: Optional[torch.Tensor] = None,
        cvar_frac: float = 0.0,
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
        device = yield_pred.device

        # 1. Yield term: weighted MSE between predicted and target yield
        per_sample_loss = torch.nn.functional.mse_loss(
            yield_pred.squeeze(), target_yield.squeeze(), reduction="none"
        )
        weights = (
            sample_weights.to(device) if sample_weights is not None else torch.ones_like(per_sample_loss)
        )

        if cvar_frac > 0.0:
            # Focus on hardest samples (top cvar_frac)
            k = max(1, int(torch.ceil(torch.tensor(len(per_sample_loss) * cvar_frac)).item()))
            topk_losses, _ = torch.topk(per_sample_loss * weights, k)
            yield_loss = topk_losses.mean()
        else:
            weighted = per_sample_loss * weights
            yield_loss = weighted.sum() / weights.sum().clamp(min=1e-8)

        beta = self._current_beta()

        # 2. Reconstruction term: Gaussian negative log-likelihood for weather features
        reconstruction_term = torch.tensor(0.0, device=device)

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
        y_past,
        target_yield,
        year,
    ) -> Dict[str, torch.Tensor]:
        """Compute variational training loss for VITA yield prediction."""
        # Forward pass through VITA model
        # Returns (yield_pred, z, mu_x, var_x, mu_p, var_p)
        padded_weather = self._apply_feature_dropout(padded_weather)
        attn_bias = self._token_attention_bias(padded_weather, weather_feature_mask)
        model_outputs = self.model(
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
            attention_bias=attn_bias,
        )

        year_values = year.squeeze()
        if year_values.dim() == 0:
            year_values = year_values.unsqueeze(0)
        # Build per-sample weights based on year mapping
        if self.year_weights:
            weight_list = [
                float(self.year_weights.get(int(y.item()), 1.0)) for y in year_values
            ]
            sample_weights = torch.tensor(weight_list, device=self.device, dtype=torch.float32)
        else:
            sample_weights = None

        # Combine with drift-based weights if enabled
        drift_weights = self._batch_drift_weights(padded_weather, weather_feature_mask)
        if drift_weights is not None:
            if sample_weights is None:
                sample_weights = drift_weights
            else:
                sample_weights = sample_weights * drift_weights

        return self.compute_elbo_loss(
            padded_weather,
            weather_feature_mask,
            target_yield,
            *model_outputs,
            sample_weights=sample_weights,
            cvar_frac=self.cvar_frac,
        )

    def compute_validation_loss(  # type: ignore
        self,
        padded_weather,
        coord_processed,
        year_expanded,
        interval,
        weather_feature_mask,
        y_past,
        target_yield,
        year,
    ) -> Dict[str, torch.Tensor]:
        """Compute variational validation loss for VITA yield prediction."""
        with torch.no_grad():
            attn_bias = self._token_attention_bias(padded_weather, weather_feature_mask)
            model_outputs = self.model(
                padded_weather,
                coord_processed,
                year_expanded,
                interval,
                weather_feature_mask,
                y_past,
                attention_bias=attn_bias,
            )
        # ### Additional code
        # # unpack model outputs
        # yield_pred = model_outputs[0].squeeze()
        # y_true = target_yield.squeeze()

        # # convert tensors → numpy
        # y_true_np = y_true.detach().cpu().numpy()
        # y_pred_np = yield_pred.detach().cpu().numpy()

        # # compute metrics
        # rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
        # r2 = float(r2_score(y_true_np, y_pred_np))

        # # Print to SLURM logs
        # print(f"\n===== TEST RESULTS FOR YEAR {self.test_year} =====")
        # print(f"RMSE: {rmse:.4f}")
        # print(f"R²:   {r2:.4f}")
        # print("===============================================\n")

        components = self.compute_elbo_loss(
            padded_weather, weather_feature_mask, target_yield, *model_outputs
        )

        # Return RMSE for validation
        return {"total_loss": components["yield"] ** 0.5}
    
    def evaluate_test_year(self):
        """Run a full test evaluation once after training finishes."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import pearsonr
        import numpy as np

        self.model.eval()

        preds_all = []
        targets_all = []

        _, test_loader = self.get_dataloaders(shuffle=False)

        with torch.no_grad():
            for batch in test_loader:
                batch = [b.to(self.device) for b in batch]

                (
                    padded_weather,
                    coord_processed,
                    year_expanded,
                    interval,
                    weather_feature_mask,
                    y_past,
                    target_yield,
                    year,
                ) = batch

                attn_bias = self._token_attention_bias(padded_weather, weather_feature_mask)
                model_outputs = self.model(
                    padded_weather,
                    coord_processed,
                    year_expanded,
                    interval,
                    weather_feature_mask,
                    y_past,
                    attention_bias=attn_bias,
                )

                preds = model_outputs[0].detach().cpu().view(-1)
                targets = target_yield.detach().cpu().view(-1)

                preds_all.append(preds)
                targets_all.append(targets)

        preds_all = np.concatenate([p.numpy() for p in preds_all])
        targets_all = np.concatenate([t.numpy() for t in targets_all])

        # ===============================
        # Compute metrics
        # ===============================

        rmse = np.sqrt(mean_squared_error(targets_all, preds_all))
        mae = mean_absolute_error(targets_all, preds_all)
        r2 = r2_score(targets_all, preds_all)

        pearson_r, _ = pearsonr(targets_all, preds_all)
        pearson_r2 = pearson_r ** 2

        mean_true = np.mean(targets_all)
        mean_pred = np.mean(preds_all)

        bias = mean_pred - mean_true
        bias2 = bias ** 2

        residuals = targets_all - preds_all
        mean_residual = np.mean(residuals)
        variance = np.mean((residuals - mean_residual) ** 2)
        std_dev = np.sqrt(variance)

        mse_verify = bias2 + variance
        rmse_verify = np.sqrt(mse_verify)

        # ===============================
        # Print results
        # ===============================

        print("\n======= FINAL TEST RESULTS (FULL DATASET) =======")
        print(f"Test Year:           {self.test_year}")
        print("-------------------------------------------------")
        print(f"R²:                  {r2:.4f}")
        print(f"Pearson R²:          {pearson_r2:.4f}")
        print(f"RMSE:                {rmse:.4f}")
        print(f"MAE:                 {mae:.4f}")
        print(f"Bias:                {bias:.4f}")
        print(f"Bias²:               {bias2:.4f}")
        print(f"Residual Variance:   {variance:.4f}")
        print(f"Residual Std Dev:    {std_dev:.4f}")
        print(f"MSE (bias²+var):     {mse_verify:.4f}")
        print(f"RMSE Verified:       {rmse_verify:.4f}")
        print("=================================================\n")

        # ===============================
        # Scatterplot
        # ===============================

        plt.figure(figsize=(6, 6))
        plt.scatter(targets_all, preds_all, alpha=0.5)
        plt.xlabel("True Yield")
        plt.ylabel("Predicted Yield")
        plt.title(f"Yield Scatterplot – Test Year {self.test_year}")
        plt.plot(
            [targets_all.min(), targets_all.max()],
            [targets_all.min(), targets_all.max()],
            'r--'
        )

        os.makedirs("logs", exist_ok=True)
        out_path = os.path.join("logs", f"scatter_test_year_{self.test_year}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"📊 Scatterplot saved to: {out_path}\n")

        return rmse, r2, bias, variance


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
        "year_weights": args_dict.get("year_weights", {}),
        "cvar_frac": args_dict.get("cvar_frac", 0.0),
        "feature_dropout_prob": args_dict.get("feature_dropout_prob", 0.0),
        "feature_dropout_protect_indices": args_dict.get(
            "feature_dropout_protect_indices", []
        ),
        "drift_weight_strength": args_dict.get("drift_weight_strength", 0.0),
        "drift_min_weight": args_dict.get("drift_min_weight", 0.2),
        "drift_features": args_dict.get("drift_features", []),
        "drift_target_year": args_dict.get("drift_target_year"),
        "attn_bias_strength": args_dict.get("attn_bias_strength", 0.0),
        "target_stats_start_week": args_dict.get("target_stats_start_week", 15),
        "target_stats_max_week": args_dict.get("target_stats_max_week", 35),
        "weather_vars": args_dict.get("weather_vars"),
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
        "year_weights": setup_params["year_weights"],
        "cvar_frac": setup_params["cvar_frac"],
        "feature_dropout_prob": setup_params["feature_dropout_prob"],
        "feature_dropout_protect_indices": setup_params["feature_dropout_protect_indices"],
        "drift_weight_strength": setup_params["drift_weight_strength"],
        "drift_min_weight": setup_params["drift_min_weight"],
        "drift_features": setup_params["drift_features"],
        "drift_target_year": setup_params["drift_target_year"],
        "attn_bias_strength": setup_params["attn_bias_strength"],
        "target_stats_start_week": setup_params["target_stats_start_week"],
        "target_stats_max_week": setup_params["target_stats_max_week"],
        "weather_vars": setup_params["weather_vars"],
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
