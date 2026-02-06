"""
Run VITA corn yield experiment on gridmet-weekly data (calendar-aligned weeks 15-44).

Defaults are CPU-friendly; override as needed:
python3 run_gridmet_corn_2012.py --pretrained-model-path checkpoints/vita_pretrained.pth
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default="",
        help="Path to pretrained VITA weights (leave empty to train from scratch)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Root data directory containing khaki_corn_belt/gridmet-weekly.csv",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (smaller for CPU)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        help="Model size (mini/small/medium/large).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-4,
        help="Beta parameter for VITA variational loss.",
    )
    parser.add_argument(
        "--init-lr",
        type=float,
        default=2.5e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--crop-type",
        type=str,
        default="corn",
        help="Crop type to predict.",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=5, help="Epochs (adjust for your machine)"
    )
    parser.add_argument(
        "--n-train-years",
        type=int,
        default=28,
        help="Years of training data (matches eval script).",
    )
    parser.add_argument(
        "--n-past-years",
        type=int,
        default=3,
        help="Past years fed to model (matches eval script).",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default="extreme",
        help="Test split type (extreme/overall/ahead_pred).",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2012,
        help="Test year used for training/eval splits.",
    )
    parser.add_argument(
        "--year-weights",
        type=str,
        default=None,
        help='JSON mapping of year to loss weight, e.g., \'{"1988":3.0,"1993":3.0,"2002":3.0}\'.',
    )
    parser.add_argument(
        "--cvar-frac",
        type=float,
        default=0.0,
        help="Fraction (0-1) of hardest weighted samples to optimize (CVaR). 0 disables.",
    )
    parser.add_argument(
        "--feature-dropout-prob",
        type=float,
        default=0.0,
        help="Probability of dropping a non-protected weather feature during training (0-1).",
    )
    parser.add_argument(
        "--feature-dropout-protect",
        type=str,
        default="7,12",
        help="Comma-separated list of weather feature indices to always keep (e.g., '7,12' for pr and vpd).",
    )
    parser.add_argument(
        "--drift-weight-strength",
        type=float,
        default=0.0,
        help="Strength of drift-based importance weights (0 disables).",
    )
    parser.add_argument(
        "--drift-min-weight",
        type=float,
        default=0.2,
        help="Minimum weight applied after drift weighting.",
    )
    parser.add_argument(
        "--drift-features",
        type=str,
        default="pr,vpd",
        help="Comma-separated weather feature names to monitor for drift (e.g., 'pr,vpd').",
    )
    parser.add_argument(
        "--weather-vars",
        type=str,
        default="all",
        help="Comma-separated gridMET weather vars to use (e.g., 'vpd' or 'pr,vpd'). Use 'all' for default.",
    )
    parser.add_argument(
        "--drift-target-year",
        type=int,
        default=None,
        help="Year to use as drift reference (defaults to test_year).",
    )
    parser.add_argument(
        "--attn-bias-strength",
        type=float,
        default=0.0,
        help="Strength of attention bias toward target-like tokens (0 disables).",
    )
    parser.add_argument(
        "--target-stats-start-week",
        type=int,
        default=15,
        help="First week to include when computing drift/attn reference stats (align to season start).",
    )
    parser.add_argument(
        "--target-stats-max-week",
        type=int,
        default=35,
        help="Use only weeks up to this number when computing drift/attn reference stats (default 35, end of Aug).",
    )
    parser.add_argument(
        "--eval-model-path",
        type=str,
        default="",
        help="Optional: run eval_vita_single_year.py on this trained model path after training.",
    )
    args = parser.parse_args()

    cli_args = [
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--model-size",
        args.model_size,
        "--beta",
        str(args.beta),
        "--init-lr",
        str(args.init_lr),
        "--test-type",
        args.test_type,
        "--crop-type",
        args.crop_type,
        "--test-year",
        str(args.test_year),
        "--n-train-years",
        str(args.n_train_years),
        "--n-past-years",
        str(args.n_past_years),
        "--data-dir",
        args.data_dir,
    ]
    if args.year_weights is not None:
        cli_args.extend(["--year-weights", args.year_weights])
    cli_args.extend(["--cvar-frac", str(args.cvar_frac)])
    cli_args.extend(["--feature-dropout-prob", str(args.feature_dropout_prob)])
    if args.feature_dropout_protect:
        cli_args.extend(["--feature-dropout-protect", args.feature_dropout_protect])
    cli_args.extend(["--drift-weight-strength", str(args.drift_weight_strength)])
    cli_args.extend(["--drift-min-weight", str(args.drift_min_weight)])
    if args.drift_features:
        cli_args.extend(["--drift-features", args.drift_features])
    if args.drift_target_year is not None:
        cli_args.extend(["--drift-target-year", str(args.drift_target_year)])
    cli_args.extend(["--attn-bias-strength", str(args.attn_bias_strength)])
    cli_args.extend(["--target-stats-start-week", str(args.target_stats_start_week)])
    cli_args.extend(["--target-stats-max-week", str(args.target_stats_max_week)])
    if args.pretrained_model_path:
        cli_args.extend(["--pretrained-model-path", args.pretrained_model_path])
    if args.weather_vars:
        cli_args.extend(["--weather-vars", args.weather_vars])

    # Delegate to the existing CLI
    from src.crop_yield.main import main as vita_main

    sys.argv = ["vitacli"] + cli_args
    vita_main()

    # Optional: run full-prediction metrics using eval_vita_single_year.py
    eval_model_path = args.eval_model_path
    if not eval_model_path:
        # Try to auto-pick most recent *_best.pth in crop_yield model dir
        import glob
        from pathlib import Path

        model_dir = Path(args.data_dir) / "trained_models" / "crop_yield"
        candidates = sorted(
            model_dir.glob("*_best.pth"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            eval_model_path = str(candidates[0])

    if eval_model_path:
        from eval_vita_single_year import main as eval_single_year_main

        eval_args = [
            "--model-path",
            eval_model_path,
            "--crop-type",
            "corn",
            "--test-year",
            str(args.test_year),
            "--n-train-years",
            str(args.n_train_years),
            "--n-past-years",
            str(args.n_past_years),
            "--model-size",
            "small",
            "--k",
            "1",
            "--batch-size",
            str(max(8, args.batch_size)),
            "--test-type",
            args.test_type,
            "--data-dir",
            args.data_dir,
        ]
        if args.weather_vars:
            eval_args.extend(["--weather-vars", args.weather_vars])
        sys.argv = ["vitaeval"] + eval_args
        eval_single_year_main()


if __name__ == "__main__":
    main()
