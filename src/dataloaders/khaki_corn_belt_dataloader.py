from asyncio.log import logger
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.utils.constants import (
    MAX_CONTEXT_LENGTH,
    TOTAL_WEATHER_VARS,
    CROP_YIELD_STATS,
)

# GridMET weekly configuration: mapped vars into 31-slot NASA layout
# Only vars with clear NASA counterparts are used
GRIDMET_VARS = [
    "pr",    # precip
    "srad",  # solar
    "tmmx",  # max temp
    "tmmn",  # min temp
    "sph",   # specific humidity
    "vs",    # wind speed
    "th",    # wind direction
    "vpd",   # vapor pressure deficit
]
N_WEEKS = 52
SEASON_START_WEEK = 15  # 1-based; weeks 15-44 contain data
SEASON_WEEKS = 30
# Map gridMET vars into 31-feature positions used by pretrained VITA (NASA POWER layout)
GRIDMET_TO_NASA_IDX = {
    "pr": 7,    # PRECTOTCORR
    "srad": 8,  # ALLSKY_SFC_SW_DWN
    "tmmx": 1,  # T2M_MAX
    "tmmn": 2,  # T2M_MIN
    "sph": 29,  # use VAP slot for specific humidity
    "vs": 3,    # wind speed -> unused slot
    "th": 4,    # wind direction -> unused slot
    "vpd": 12,  # vapor pressure deficit -> unused slot
}


def resolve_weather_vars(weather_vars: list[str] | None) -> list[str]:
    if not weather_vars:
        return GRIDMET_VARS
    unknown = [v for v in weather_vars if v not in GRIDMET_TO_NASA_IDX]
    if unknown:
        raise ValueError(
            f"Unknown weather vars: {unknown}. Allowed: {sorted(GRIDMET_TO_NASA_IDX.keys())}"
        )
    return weather_vars


def standardize_weather_cols(
    data: pd.DataFrame, weather_vars: list[str] | None = None
) -> pd.DataFrame:
    """
    Standardize only weather columns using dataset-based scalers.

    Args:
        data: DataFrame containing weather data

    Returns:
        DataFrame with standardized weather columns only
    """
    data_copy = data.copy()

    # Get weather columns (weekly gridMET vars)
    weather_vars = resolve_weather_vars(weather_vars)
    weather_cols = [
        col for col in data_copy.columns if any(col.startswith(f"{v}_week_") for v in weather_vars)
    ]
    weather_cols_in_data = [col for col in weather_cols if col in data_copy.columns]

    # Use dataset-based scalers for weather columns only
    if weather_cols_in_data:
        means = data_copy[weather_cols_in_data].mean()
        stds = data_copy[weather_cols_in_data].std()
        data_copy[weather_cols_in_data] = (
            data_copy[weather_cols_in_data] - means
        ) / stds
        # Fill any NaN values that result from division by zero with 0
        data_copy[weather_cols_in_data] = data_copy[weather_cols_in_data].fillna(0)

    return data_copy


class CropDataset(Dataset):
    def __init__(
        self,
        data,
        start_year,
        test_year,
        test_dataset=False,
        n_past_years=5,
        test_gap=0,
        crop_type="soybean",
        weather_vars: list[str] | None = None,
    ):
        self.crop_type = crop_type
        self.yield_col = f"{crop_type}_yield"

        self.weather_vars = resolve_weather_vars(weather_vars)
        self.weather_cols = [
            f"{v}_week_{j}" for v in self.weather_vars for j in range(1, N_WEEKS + 1)
        ]
        self.practice_cols: list[str] = []
        self.soil_cols: list[str] = []

        # Define weather indices used in preprocessing (mapped into 31-slot layout)
        self.weather_indices = torch.tensor(
            [GRIDMET_TO_NASA_IDX[v] for v in self.weather_vars]
        )
        # substract test gap from start year
        start_year -= test_gap

        if test_dataset:  # test on specific year
            candidate_data = data[data["year"] == test_year]
        else:  # train on years from start_year to year before test_year - test_gap
            candidate_data = data[
                (data["year"] >= start_year) & (data["year"] < test_year - test_gap)
            ]

        # Filter to only include cases where we have complete historical data
        # Vectorized approach: group by loc_ID and check consecutive years availability
        data_sorted = data.sort_values(["loc_ID", "year"])

        # For each candidate, check if we can get exactly n_past_years + 1 consecutive years ending at that year
        def has_sufficient_history(row):
            year, loc_ID = row["year"], row["loc_ID"]
            loc_data = data_sorted[data_sorted["loc_ID"] == loc_ID]
            loc_data_up_to_year = loc_data[loc_data["year"] <= year]
            return len(loc_data_up_to_year.tail(n_past_years + 1)) == n_past_years + 1

        # Apply vectorized check
        mask = candidate_data.apply(has_sufficient_history, axis=1)
        valid_candidates = candidate_data[mask]

        self.index = valid_candidates[["year", "loc_ID"]].reset_index(drop=True)

        dataset_name = "train" if not test_dataset else "test"
        logger.info(
            f"Creating {dataset_name} dataloader with {len(self.index)} samples for {'test year ' + str(test_year) if test_dataset else 'training years ' + str(start_year) + '-' + str(test_year-test_gap-1)} using {crop_type} yield."
        )

        self.data = []
        total_samples = len(self.index)
        samples_to_process = total_samples

        if total_samples == 0:
            logger.warning(f"No samples found for {dataset_name} dataset!")
            return

        for idx in range(min(samples_to_process, total_samples)):
            year, loc_ID = self.index.iloc[idx].values.astype("int")
            # Get exactly n_past_years + 1 years of data for this location
            query_data = data[(data["year"] <= year) & (data["loc_ID"] == loc_ID)].tail(
                n_past_years + 1
            )

            weather = (
                query_data[self.weather_cols]
                .values.astype("float32")
                .reshape((-1, len(self.weather_vars), N_WEEKS))
            )  # 8 measurements, 52 weeks (season weeks 15-44 populated)
            n_years = weather.shape[0]
            year_data = query_data["year"].values.astype("float32")
            coord = torch.FloatTensor(
                query_data[["lat", "lng"]].values.astype("float32")
            )

            # get the true yield
            y = query_data.iloc[-1:][self.yield_col].values.astype("float32").copy()
            y_past = query_data[self.yield_col].values.astype("float32")
            if len(y_past) <= 1:
                raise ValueError(
                    f"Only 1 year of yield data for location {loc_ID} in year {year}. "
                )
            # the current year's yield is the target variable, so replace it with last year's yield
            # this is done to follow Khaki et. al.
            y_past[-1] = y_past[-2]

            # Preprocess weather data for the model
            n_years, n_features, seq_len = weather.shape

            # Check context length constraint
            if n_years * seq_len > MAX_CONTEXT_LENGTH:
                raise ValueError(
                    f"n_years * seq_len = {n_years * seq_len} is greater than MAX_CONTEXT_LENGTH = {MAX_CONTEXT_LENGTH}"
                )

            # Transpose and reshape weather data: (n_years, n_features, seq_len) -> (n_years * seq_len, n_features)
            weather = weather.transpose(0, 2, 1)  # (n_years, seq_len, n_features)
            weather = weather.reshape(
                n_years * seq_len, n_features
            )  # (n_years * seq_len, n_features)

            # Process coordinates - use only the first coordinate (same for all years in this location)
            coord_processed = coord[0, :]  # (2,)

            # Expand year to match the sequence length
            # year_data is [n_years], need to add fraction for each week (1/52, 2/52, ..., 52/52)
            week_fractions = (
                torch.arange(1, seq_len + 1, dtype=torch.float32) / seq_len
            )  # [seq_len]
            year_expanded = torch.FloatTensor(year_data).unsqueeze(
                1
            ) + week_fractions.unsqueeze(  # [n_years, 1]
                0
            )  # [1, seq_len]  # [n_years, seq_len]
            year_expanded = year_expanded.contiguous().view(
                n_years * seq_len
            )  # [n_years * seq_len]

            # Create padded weather with specific weather indices (already 52 weeks)
            padded_steps = n_years * seq_len
            padded_weather = torch.zeros(
                (padded_steps, TOTAL_WEATHER_VARS),
            )
            padded_weather[:, self.weather_indices] = torch.FloatTensor(weather)

            # Create weather feature mask: unmask only season weeks (15-44) for each year
            weather_feature_mask = torch.ones(
                padded_steps,
                TOTAL_WEATHER_VARS,
                dtype=torch.bool,
            )
            season_start = SEASON_START_WEEK - 1
            season_end = season_start + SEASON_WEEKS
            for y_idx in range(n_years):
                start = y_idx * seq_len + season_start
                end = y_idx * seq_len + season_end
                weather_feature_mask[start:end, self.weather_indices] = False

            # Create temporal interval (weekly data)
            interval = torch.full((1,), 7, dtype=torch.float32)

            # Keep year for loss weighting / analysis
            year_tensor = torch.tensor([year], dtype=torch.float32)

            self.data.append(
                (
                    padded_weather,  # (n_years * 52, TOTAL_WEATHER_VARS)
                    coord_processed,  # (2,)
                    year_expanded,  # (n_years * 52,)
                    interval,  # (1,)
                    weather_feature_mask,  # (n_years * 52, TOTAL_WEATHER_VARS)
                    y_past,  # (n_years,)
                    y,  # (1,)
                    year_tensor,  # (1,)
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data_loader(self, batch_size=32, shuffle=False, num_workers=4):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


def split_train_test_by_year(
    soybean_df: pd.DataFrame,
    n_train_years: int,
    test_year: int,
    standardize: bool,
    n_past_years: int,
    crop_type: str,
    test_gap: int = 0,
    weather_vars: list[str] | None = None,
):
    # you need n_train_years + 1 years of data
    # n_train years to have at least one training datapoint
    # last 1 year is test year
    start_year = test_year - n_train_years

    data = soybean_df[
        soybean_df["year"] > 1981.0
    ].copy()  # must be > 1981 otherwise all past data is just 0

    yield_col = f"{crop_type}_yield"

    # Drop rows with missing yield values for the given crop before standardization
    rows_before = len(data)
    data = data.dropna(subset=[yield_col])  # type: ignore
    rows_after = len(data)
    rows_dropped = rows_before - rows_after

    if rows_dropped > 0:
        logger.warning(
            f"Dropped {rows_dropped} rows with missing {crop_type} yield values ({rows_before} -> {rows_after} rows)"
        )

    data = data.fillna(0)

    if standardize:
        # First standardize weather data
        data = standardize_weather_cols(data, weather_vars=weather_vars)

        # Then standardize non-weather data using original approach
        cols_to_exclude = [
            "loc_ID",
            "year",
            "State",
            "County",
            "lat",
            "lng",
            yield_col,
            "geometry",
            "centroid",
            "STATEFP",
            "COUNTYFP",
        ]
        # Also exclude weather columns since we already standardized them
        weather_vars = resolve_weather_vars(weather_vars)
        weather_cols = [
            f"{v}_week_{j}" for v in weather_vars for j in range(1, N_WEEKS + 1)
        ]
        cols_to_exclude.extend(weather_cols)

        cols_to_standardize = [
            col
            for col in data.columns
            if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(data[col])
        ]

        # Standardize non-weather data (soil, practices, etc.)
        if cols_to_standardize:
            data[cols_to_standardize] = (
                data[cols_to_standardize] - data[cols_to_standardize].mean()
            ) / data[cols_to_standardize].std()
            # Fill any NaN values that result from division by zero with 0
            data[cols_to_standardize] = data[cols_to_standardize].fillna(0)

        # save crop-specific yield statistics from constants
        train_data = data[(data["year"] >= start_year) & (data["year"] < test_year)]
        yield_mean, yield_std = (
            train_data[yield_col].mean(),
            train_data[yield_col].std(),
        )
        data[yield_col] = (data[yield_col] - yield_mean) / yield_std
        logger.info(
            f"Saving mean ({yield_mean:.3f}) and std ({yield_std:.3f}) from training data for {crop_type}"
        )
        CROP_YIELD_STATS[crop_type]["mean"].append(yield_mean)
        CROP_YIELD_STATS[crop_type]["std"].append(yield_std)

    train_dataset = CropDataset(
        data.copy(),
        start_year,
        test_year,
        test_dataset=False,
        n_past_years=n_past_years,
        test_gap=test_gap,
        crop_type=crop_type,
        weather_vars=weather_vars,
    )
    test_dataset = CropDataset(
        data.copy(),
        start_year,
        test_year,
        test_dataset=True,
        n_past_years=n_past_years,
        test_gap=test_gap,
        crop_type=crop_type,
        weather_vars=weather_vars,
    )

    # Return the train and test datasets
    return train_dataset, test_dataset


def read_khaki_corn_belt_dataset(data_dir: str):
    # Use weekly aggregated gridMET features
    full_filename = "khaki_corn_belt/gridmet-weekly.csv"
    usa_df = pd.read_csv(data_dir + full_filename)
    usa_df = usa_df.sort_values(["loc_ID", "year"])
    return usa_df


def get_train_test_loaders(
    crop_df: pd.DataFrame,
    n_train_years: int,
    test_year: int,
    n_past_years: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    crop_type: str,
    test_gap: int = 0,
    weather_vars: list[str] | None = None,
) -> Tuple[DataLoader, DataLoader]:

    if n_train_years <= 1:
        raise ValueError(
            f"Not enough training data for current year + n_past_years. Required: {n_past_years + 1}. "
            f"Available training years: {n_train_years}."
        )

    if n_train_years < n_past_years + 1:
        logger.warning(
            f"Not enough training data for current year + n_past_years. Required: {n_past_years + 1}. "
            f"Available training years: {n_train_years}. "
            f"Setting n_past_years to {n_train_years - 1}."
        )
        n_past_years = n_train_years - 1

    train_dataset, test_dataset = split_train_test_by_year(
        crop_df,
        n_train_years,
        test_year,
        standardize=True,
        n_past_years=n_past_years,
        crop_type=crop_type,
        test_gap=test_gap,
        weather_vars=weather_vars,
    )

    if n_train_years < n_past_years + 1:
        raise ValueError(
            f"Not enough training data for current year + n_past_years. Required: {n_past_years + 1}. "
            f"Available training years: {n_train_years}."
        )

    train_loader = train_dataset.get_data_loader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = test_dataset.get_data_loader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader
