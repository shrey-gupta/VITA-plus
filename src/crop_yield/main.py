import argparse
import logging
import os
import random
import json
from typing import List

import numpy as np
import torch

from src.crop_yield.vita_yield_trainer import vita_yield_training_loop
from src.utils.utils import setup_logging, parse_args
from src.utils.constants import CROP_YIELD_STATS, DEFAULT_DATA_DIR
from src.dataloaders.khaki_corn_belt_dataloader import GRIDMET_TO_NASA_IDX

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-dir",
    help="path to data directory",
    default=DEFAULT_DATA_DIR,
    type=str,
)
parser.add_argument("--batch-size", help="batch size", default=64, type=int)
parser.add_argument(
    "--n-past-years", help="number of past years to look at", default=6, type=int
)
parser.add_argument(
    "--n-epochs", help="number of training epochs", default=40, type=int
)
parser.add_argument(
    "--init-lr", help="initial learning rate for Adam", default=0.0005, type=float
)
parser.add_argument(
    "--decay-factor",
    help="learning rate exponential decay factor",
    default=None,
    type=float,
)
parser.add_argument(
    "--n-warmup-epochs", help="number of warmup epochs", default=10, type=int
)
parser.add_argument(
    "--pretrained-model-path",
    help="path to pretrained model weights",
    required=True,
    type=str,
)
parser.add_argument(
    "--model-size",
    help="model size mini (60k) small (2M), medium (8M), and large (56M)",
    default="small",
    type=str,
)
parser.add_argument(
    "--n-train-years",
    help="number of years of training data to use (start year will be calculated as test_year - n_train_years + 1)",
    default=15,
    type=int,
)
parser.add_argument(
    "--k",
    help="number of sinusoidal components for VITA",
    default=1,
    type=int,
)
parser.add_argument(
    "--beta",
    help="beta parameter for VITA variational loss",
    default=1e-4,
    type=float,
)

parser.add_argument(
    "--seed",
    help="seed for random number generator",
    default=1234,
    type=int,
)
parser.add_argument(
    "--crop-type",
    help="crop type to predict: soybean, corn, wheat, sunflower, cotton, sugarcane, beans, corn_rainfed, beans_rainfed",
    default="soybean",
    type=str,
    choices=[
        "soybean",
        "corn",
    ],
)
parser.add_argument(
    "--test-year",
    help="specific test year for single-year evaluation (if not provided, uses 5-fold cross validation)",
    default=None,
    type=int,
)
parser.add_argument(
    "--test-type",
    help="type of test evaluation: extreme (extreme years), overall (2014-18), or ahead_pred (2014-18 with 5-year gap)",
    default="extreme",
    type=str,
    choices=["extreme", "overall", "ahead_pred"],
)
parser.add_argument(
    "--resume-from-checkpoint",
    help="path to resume from checkpoint",
    default=None,
    type=str,
)
parser.add_argument(
    "--year-weights",
    help='JSON mapping of year to loss weight, e.g., \'{"1988":3.0,"1993":3.0,"2002":3.0}\' (default applies these weights)',
    default=None,
    type=str,
)
parser.add_argument(
    "--cvar-frac",
    help="Fraction (0-1) of hardest weighted samples to optimize (CVaR). 0 disables.",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--feature-dropout-prob",
    help="Probability of dropping a non-protected weather feature during training (0-1).",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--feature-dropout-protect",
    help="Comma-separated list of weather feature indices to always keep (e.g., '7,12' for pr and vpd).",
    default="7,12",
    type=str,
)
parser.add_argument(
    "--drift-weight-strength",
    help="Strength of drift-based importance weights (0 disables).",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--drift-min-weight",
    help="Minimum weight applied after drift weighting to avoid zeroing samples.",
    default=0.2,
    type=float,
)
parser.add_argument(
    "--drift-features",
    help="Comma-separated weather feature names to monitor for drift (e.g., 'pr,vpd').",
    default="pr,vpd",
    type=str,
)
parser.add_argument(
    "--weather-vars",
    help="Comma-separated gridMET weather vars to use (e.g., 'vpd' or 'pr,vpd'). Use 'all' for default.",
    default="all",
    type=str,
)
parser.add_argument(
    "--drift-target-year",
    help="Year to use as drift reference (defaults to test_year).",
    default=None,
    type=int,
)
parser.add_argument(
    "--attn-bias-strength",
    help="Strength of attention bias toward target-like tokens (0 disables).",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--target-stats-start-week",
    help="First week to include when computing drift/attn reference stats (align to season start).",
    default=15,
    type=int,
)
parser.add_argument(
    "--target-stats-max-week",
    help="Use only weeks up to this number when computing drift/attn reference stats (default 35, end of Aug).",
    default=35,
    type=int,
)


def main(args_dict=None):
    setup_logging(rank=0)

    if args_dict is None:
        args_dict = parse_args(parser)

    seed = args_dict["seed"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Parse year weights (allow default extreme emphasis)
    default_year_weights = {"1988": 3.0, "1993": 3.0, "2002": 3.0}
    year_weights_raw = args_dict.get("year_weights")
    if year_weights_raw is None:
        year_weights = default_year_weights
    else:
        try:
            year_weights = json.loads(year_weights_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse --year-weights as JSON. Got: {year_weights_raw}"
            ) from exc
    # Normalize keys to int
    args_dict["year_weights"] = {int(k): float(v) for k, v in year_weights.items()}
    args_dict["cvar_frac"] = float(args_dict.get("cvar_frac", 0.0))

    # Feature dropout protection list
    protect_raw = args_dict.get("feature_dropout_protect", "")
    protect_indices: List[int] = []
    if protect_raw:
        for tok in protect_raw.split(","):
            tok = tok.strip()
            if tok:
                protect_indices.append(int(tok))
    args_dict["feature_dropout_protect_indices"] = protect_indices
    args_dict["feature_dropout_prob"] = float(args_dict.get("feature_dropout_prob", 0.0))

    # Drift weighting config
    drift_feats_raw = args_dict.get("drift_features", "")
    drift_features: List[str] = []
    if drift_feats_raw:
        for tok in drift_feats_raw.split(","):
            tok = tok.strip()
            if tok:
                drift_features.append(tok)
    args_dict["drift_features"] = drift_features
    args_dict["drift_weight_strength"] = float(args_dict.get("drift_weight_strength", 0.0))
    args_dict["drift_min_weight"] = float(args_dict.get("drift_min_weight", 0.2))
    args_dict["drift_target_year"] = args_dict.get("drift_target_year")
    args_dict["attn_bias_strength"] = float(args_dict.get("attn_bias_strength", 0.0))
    args_dict["target_stats_start_week"] = int(args_dict.get("target_stats_start_week", 15))
    args_dict["target_stats_max_week"] = int(args_dict.get("target_stats_max_week", 35))

    weather_vars_raw = args_dict.get("weather_vars", "all")
    if weather_vars_raw is None or str(weather_vars_raw).strip().lower() == "all":
        args_dict["weather_vars"] = None
    else:
        weather_vars = [tok.strip() for tok in str(weather_vars_raw).split(",") if tok.strip()]
        unknown = [v for v in weather_vars if v not in GRIDMET_TO_NASA_IDX]
        if unknown:
            raise ValueError(
                f"Unknown weather vars: {unknown}. Allowed: {sorted(GRIDMET_TO_NASA_IDX.keys())}"
            )
        args_dict["weather_vars"] = weather_vars

    if args_dict["n_train_years"] < args_dict["n_past_years"] + 1:
        logging.warning(
            f"Not enough training data for current year + n_past_years. Required: {args_dict['n_past_years'] + 1}. "
            f"Available training years: {args_dict['n_train_years']}. "
            f"Setting n_past_years to {args_dict['n_train_years'] - 1}."
        )
        args_dict["n_past_years"] = args_dict["n_train_years"] - 1

    cross_validation_results = vita_yield_training_loop(args_dict)

    logger = logging.getLogger(__name__)
    logger.info("Training completed successfully!")

    kfold_results = cross_validation_results["fold_results"]
    fold_stds = CROP_YIELD_STATS[args_dict["crop_type"]]["std"]

    # Compute RMSE in bu/acre
    best_rmse_bu_acre = [result * std for result, std in zip(kfold_results, fold_stds)]
    avg_best_rmse = float(np.mean(best_rmse_bu_acre))
    std_best_rmse = float(np.std(best_rmse_bu_acre))

    # Compute R² for each fold: R² = 1 - (RMSE/std)²
    r_squared_values = [
        1 - (rmse / std) ** 2 for rmse, std in zip(best_rmse_bu_acre, fold_stds)
    ]
    avg_r_squared = float(np.mean(r_squared_values))
    std_r_squared = float(np.std(r_squared_values))

    logger.info(
        f"Final average best RMSE for {args_dict['crop_type']}: {avg_best_rmse:.3f} ± {std_best_rmse:.3f}"
    )
    logger.info(
        f"Final average R² for {args_dict['crop_type']}: {avg_r_squared:.3f} ± {std_r_squared:.3f}"
    )

    return avg_best_rmse, std_best_rmse, avg_r_squared, std_r_squared, r_squared_values


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed with error: {e}")
        raise
