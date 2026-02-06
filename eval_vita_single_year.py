import argparse
import logging
import os
import random
from typing import Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.crop_yield.vita_yield_model import VITAYieldModel
from src.dataloaders.khaki_corn_belt_dataloader import (
    get_train_test_loaders,
    read_khaki_corn_belt_dataset,
    N_WEEKS,
    GRIDMET_TO_NASA_IDX,
)
from src.utils.constants import CROP_YIELD_STATS, DEFAULT_DATA_DIR, TOTAL_WEATHER_VARS
from src.utils.utils import get_model_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VITA on a single test year using full predictions (apples-to-apples metrics)."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the saved model (e.g., data/trained_models/crop_yield/<model>_best.pth)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--crop-type",
        default="soybean",
        choices=["soybean", "corn"],
        help="Crop type used for training/eval.",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        required=True,
        help="Test year to evaluate (e.g., 2012).",
    )
    parser.add_argument(
        "--n-train-years",
        type=int,
        required=True,
        help="Number of past years used for training (must match the training run).",
    )
    parser.add_argument(
        "--n-past-years",
        type=int,
        required=True,
        help="Number of past years fed to the model (must match the training run).",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        choices=["mini", "small", "medium", "large"],
        help="Model size used during training (must match the checkpoint).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of sinusoidal components (must match the checkpoint).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation dataloader.",
    )
    parser.add_argument(
        "--test-type",
        default="extreme",
        choices=["extreme", "overall", "ahead_pred"],
        help="Test split type (must match how you trained).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (cpu or cuda). Default: auto-detect.",
    )
    parser.add_argument(
        "--weather-vars",
        default="all",
        help="Comma-separated gridMET weather vars to use (e.g., 'vpd' or 'pr,vpd'). Use 'all' for default.",
    )
    parser.add_argument(
        "--plot-attention",
        action="store_true",
        help="Plot average attention weights for the test year weeks.",
    )
    parser.add_argument(
        "--attention-out",
        default="logs/attention_test_year.png",
        help="Output path for attention plot (default: logs/attention_test_year.png).",
    )
    parser.add_argument(
        "--attention-csv",
        default="logs/attention_test_year.csv",
        help="Output path for attention CSV (default: logs/attention_test_year.csv).",
    )
    parser.add_argument(
        "--attention-renorm",
        action="store_true",
        help="Renormalize attention weights within the test-year weeks only.",
    )
    parser.add_argument(
        "--plot-attention-heatmap",
        action="store_true",
        help="Plot attention heatmap across all years in the input window.",
    )
    parser.add_argument(
        "--attention-heatmap-out",
        default="logs/attention_heatmap.png",
        help="Output path for attention heatmap (default: logs/attention_heatmap.png).",
    )
    return parser.parse_args()


def set_seeds(seed: int = 1234):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_model(
    model_path: str, device: torch.device, model_kwargs: dict
) -> VITAYieldModel:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, VITAYieldModel):
        model = checkpoint
        model.to(device)
        return model

    model = VITAYieldModel(device=device, **model_kwargs)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        return model

    # Fallback: assume full model object
    model = checkpoint.to(device)
    return model


def build_dataloaders(
    crop_type: str,
    data_dir: str,
    test_year: int,
    n_train_years: int,
    n_past_years: int,
    batch_size: int,
    test_type: str,
    weather_vars: list[str] | None,
):
    crop_df = read_khaki_corn_belt_dataset(data_dir)
    test_gap = 4 if test_type == "ahead_pred" else 0
    train_loader, test_loader = get_train_test_loaders(
        crop_df=crop_df,
        n_train_years=n_train_years,
        test_year=test_year,
        n_past_years=n_past_years,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        crop_type=crop_type,
        test_gap=test_gap,
        weather_vars=weather_vars,
    )
    return train_loader, test_loader


def denormalization_stats(crop_type: str) -> Tuple[float, float]:
    """Get the most recent mean/std pushed by get_train_test_loaders."""
    means = CROP_YIELD_STATS[crop_type]["mean"]
    stds = CROP_YIELD_STATS[crop_type]["std"]
    if not means or not stds:
        raise ValueError("Normalization stats not populated; run dataloader creation first.")
    return means[-1], stds[-1]


def ensure_parent_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def collect_attention_weights(
    model: VITAYieldModel,
    test_loader,
    device: torch.device,
    renorm_within_year: bool,
) -> np.ndarray:
    model.eval()
    attention_sum = np.zeros((N_WEEKS,), dtype=np.float64)
    n_samples = 0

    with torch.no_grad():
        for (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
            _,
            _,
        ) in test_loader:
            model_inputs = [
                padded_weather.to(device),
                coord_processed.to(device),
                year_expanded.to(device),
                interval.to(device),
                weather_feature_mask.to(device),
                y_past.to(device),
            ]
            outputs = model(*model_inputs, return_attention=True)
            attention_weights = outputs[-1].squeeze(-1).cpu().numpy()

            n_years = y_past.shape[1]
            seq_len = attention_weights.shape[1] // n_years
            attention_weights = attention_weights.reshape(-1, n_years, seq_len)

            attn_year = attention_weights[:, -1, :]

            if renorm_within_year:
                denom = attn_year.sum(axis=1, keepdims=True)
                denom[denom == 0] = 1.0
                attn_year = attn_year / denom

            attention_sum += attn_year.sum(axis=0)
            n_samples += attn_year.shape[0]

    if n_samples == 0:
        raise ValueError("No samples found while collecting attention weights.")

    return (attention_sum / n_samples).astype(np.float32)


def collect_attention_weights_by_year(
    model: VITAYieldModel,
    test_loader,
    device: torch.device,
    renorm_within_year: bool,
) -> np.ndarray:
    model.eval()
    attention_sum = None
    n_samples = 0

    with torch.no_grad():
        for (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
            _,
            _,
        ) in test_loader:
            model_inputs = [
                padded_weather.to(device),
                coord_processed.to(device),
                year_expanded.to(device),
                interval.to(device),
                weather_feature_mask.to(device),
                y_past.to(device),
            ]
            outputs = model(*model_inputs, return_attention=True)
            attention_weights = outputs[-1].squeeze(-1).cpu().numpy()

            n_years = y_past.shape[1]
            seq_len = attention_weights.shape[1] // n_years
            attention_weights = attention_weights.reshape(-1, n_years, seq_len)

            if renorm_within_year:
                denom = attention_weights.sum(axis=2, keepdims=True)
                denom[denom == 0] = 1.0
                attention_weights = attention_weights / denom

            if attention_sum is None:
                attention_sum = attention_weights.sum(axis=0)
            else:
                attention_sum += attention_weights.sum(axis=0)
            n_samples += attention_weights.shape[0]

    if n_samples == 0 or attention_sum is None:
        raise ValueError("No samples found while collecting attention weights.")

    return (attention_sum / n_samples).astype(np.float32)


def plot_attention(attention_weights: np.ndarray, out_path: str, title: str):
    weeks = np.arange(1, len(attention_weights) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(weeks, attention_weights, linewidth=2)
    plt.xlabel("Week")
    plt.ylabel("Average attention weight")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_attention_heatmap(attention_matrix: np.ndarray, years: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(10, 4 + 0.25 * len(years)))
    plt.imshow(attention_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Average attention weight")
    plt.xlabel("Week")
    plt.ylabel("Year")
    plt.title(title)
    plt.xticks(
        ticks=np.arange(0, attention_matrix.shape[1], 4),
        labels=np.arange(1, attention_matrix.shape[1] + 1, 4),
    )
    plt.yticks(ticks=np.arange(len(years)), labels=years)
    plt.tight_layout()
    ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()


def evaluate(model: VITAYieldModel, test_loader, device, crop_type: str):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for (
            padded_weather,
            coord_processed,
            year_expanded,
            interval,
            weather_feature_mask,
            y_past,
            target_yield,
            year,
        ) in test_loader:
            model_inputs = [
                padded_weather.to(device),
                coord_processed.to(device),
                year_expanded.to(device),
                interval.to(device),
                weather_feature_mask.to(device),
                y_past.to(device),
            ]
            yield_pred, *_ = model(*model_inputs)
            preds.append(yield_pred.detach().cpu().view(-1))
            targets.append(target_yield.detach().cpu().view(-1))

    if not preds:
        raise ValueError("No predictions were generated; check the test loader.")

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()

    mean, std = denormalization_stats(crop_type)
    y_pred = y_pred * std + mean
    y_true = y_true * std + mean

    residuals = y_true - y_pred
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(y_pred.mean() - y_true.mean())
    variance = float(np.mean((residuals - residuals.mean()) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - (np.sum(residuals**2) / sst)) if sst > 0 else float("nan")

    if len(y_true) > 1:
        corr_matrix = np.corrcoef(y_true, y_pred)
        pearson_r = float(corr_matrix[0, 1])
        pearson_r2 = float(pearson_r**2)
    else:
        pearson_r = float("nan")
        pearson_r2 = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "pearson_r2": pearson_r2,
        "bias": bias,
        "variance": variance,
        "n_samples": len(y_true),
    }


def main():
    args = parse_args()
    set_seeds()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info(f"Using device: {device}")

    model_size_params = get_model_params(args.model_size)
    model_kwargs = {
        "name": "vita_eval",
        "device": device,
        "k": args.k,
        "weather_dim": TOTAL_WEATHER_VARS,
        "n_past_years": args.n_past_years,
        **model_size_params,
    }

    # Build loaders (populates CROP_YIELD_STATS with mean/std for this split)
    weather_vars = None
    if args.weather_vars is not None and str(args.weather_vars).strip().lower() != "all":
        weather_vars = [tok.strip() for tok in args.weather_vars.split(",") if tok.strip()]
        unknown = [v for v in weather_vars if v not in GRIDMET_TO_NASA_IDX]
        if unknown:
            raise ValueError(
                f"Unknown weather vars: {unknown}. Allowed: {sorted(GRIDMET_TO_NASA_IDX.keys())}"
            )
    _, test_loader = build_dataloaders(
        crop_type=args.crop_type,
        data_dir=args.data_dir,
        test_year=args.test_year,
        n_train_years=args.n_train_years,
        n_past_years=args.n_past_years,
        batch_size=args.batch_size,
        test_type=args.test_type,
        weather_vars=weather_vars,
    )

    model = load_model(args.model_path, device, model_kwargs)

    metrics = evaluate(model, test_loader, device, args.crop_type)

    print("=== VITA full-prediction metrics ===")
    for key, val in metrics.items():
        print(f"{key.upper()}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

    if args.plot_attention:
        attention_weights = collect_attention_weights(
            model,
            test_loader,
            device,
            renorm_within_year=args.attention_renorm,
        )
        title = f"Average attention by week (test year {args.test_year})"
        plot_attention(attention_weights, args.attention_out, title)
        ensure_parent_dir(args.attention_csv)
        np.savetxt(
            args.attention_csv,
            np.column_stack((np.arange(1, N_WEEKS + 1), attention_weights)),
            delimiter=",",
            header="week,attention_weight",
            comments="",
        )
        print(f"Saved attention plot to {args.attention_out}")
        print(f"Saved attention CSV to {args.attention_csv}")

    if args.plot_attention_heatmap:
        attention_matrix = collect_attention_weights_by_year(
            model,
            test_loader,
            device,
            renorm_within_year=args.attention_renorm,
        )
        years = np.arange(args.test_year - args.n_past_years, args.test_year + 1)
        title = f"Attention heatmap by week (window ending {args.test_year})"
        plot_attention_heatmap(attention_matrix, years, args.attention_heatmap_out, title)
        print(f"Saved attention heatmap to {args.attention_heatmap_out}")


if __name__ == "__main__":
    main()
