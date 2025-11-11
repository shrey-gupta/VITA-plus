import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.base.base_model import BaseModel
from src.base.vanilla_pos_encoding import VanillaPositionalEncoding
from src.utils.constants import MAX_CONTEXT_LENGTH, DEVICE
from src.utils.utils import normalize_year_interval_coords


class VITA(BaseModel):
    def __init__(
        self,
        weather_dim,
        output_dim,
        k=1,  # number of seasonal components
        num_heads=20,
        num_layers=8,
        hidden_dim_factor=24,
        max_len=MAX_CONTEXT_LENGTH,
        device=DEVICE,
    ):
        super(VITA, self).__init__("vita")

        self.weather_dim = weather_dim
        self.input_dim = weather_dim + 1 + 2  # weather + (year-1970)/100 + coordinates
        self.output_dim = output_dim
        self.max_len = max_len

        hidden_dim = hidden_dim_factor * num_heads
        feedforward_dim = hidden_dim * 4

        self.in_proj = nn.Linear(self.input_dim, hidden_dim)

        self.positional_encoding = VanillaPositionalEncoding(
            hidden_dim, max_len=max_len, device=device
        )
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            device=device,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.out_proj = nn.Linear(hidden_dim, 2 * output_dim)

        self.positions = torch.arange(
            max_len, dtype=torch.float, device=device
        ).reshape(
            1, 1, max_len, 1
        )  #
        self.k = k

        # Initialize with shape (1, k, max_len, weather_dim) to avoid unsqueezing later
        self.frequency = nn.Parameter(torch.randn(1, k, max_len, weather_dim) * 0.1)
        self.phase = nn.Parameter(torch.randn(1, k, max_len, weather_dim) * 0.1)
        self.amplitude = nn.Parameter(torch.randn(1, k, max_len, weather_dim) * 0.1)
        self.log_var_prior = nn.Parameter(
            torch.randn(1, max_len, weather_dim) * 0.1 - 1
        )

    def load_pretrained(self, pretrained_model: "VITA"):  # type: ignore
        """Load weights from a pretrained VITA model by deep copying each layer."""

        if self.input_dim != pretrained_model.input_dim:
            raise ValueError(
                f"expected input dimension {self.input_dim} but received {pretrained_model.input_dim}"
            )
        if self.max_len != pretrained_model.max_len:
            raise ValueError(
                f"expected max length {self.max_len} but received {pretrained_model.max_len}"
            )

        self.in_proj = copy.deepcopy(pretrained_model.in_proj)
        self.positional_encoding = copy.deepcopy(pretrained_model.positional_encoding)
        self.transformer_encoder = copy.deepcopy(pretrained_model.transformer_encoder)
        self.out_proj = copy.deepcopy(pretrained_model.out_proj)

    def transformer_forward(
        self,
        input_tensor: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self.in_proj(input_tensor)
        input_tensor = self.positional_encoding(input_tensor)
        input_tensor = self.transformer_encoder(
            input_tensor, src_key_padding_mask=src_key_padding_mask
        )
        output = self.out_proj(input_tensor)

        # Split output into mu and log_var (VAE-style parameterization)
        mu_x = output[..., : self.output_dim]
        log_var_x = output[..., self.output_dim :]
        var_x = torch.exp(log_var_x)
        return mu_x, var_x

    def sinusoidal_prior_forward(
        self,
        weather: torch.Tensor,
        interval: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the actual sequence length from the input
        seq_len = weather.shape[1]
        batch_size = weather.shape[0]

        # Compute sinusoidal prior: p(z) ~ N(A * sin(theta * pos + phase), sigma^2_p)
        # Parameters are already shaped as (1, k, max_len, weather_dim)
        amplitude = self.amplitude[:, :, :seq_len, :]  # (1, k, seq_len, weather_dim)
        phase = self.phase[:, :, :seq_len, :]  # (1, k, seq_len, weather_dim)
        frequency = self.frequency[:, :, :seq_len, :]  # (1, k, seq_len, weather_dim)

        # pos is (1, seq_len, 1)
        pos = self.positions[:, :, :seq_len, :]
        # scaled_pos is (1, 1, seq_len, 1) -> (batch_size, 1, seq_len, 1)
        scaled_pos = pos * 2 * torch.pi * interval.view(batch_size, 1, 1, 1) / 365.0

        # Now broadcasting works directly: (batch_size, k, seq_len, weather_dim)
        sines = amplitude * torch.sin(frequency * scaled_pos + phase)
        mu_p = torch.sum(
            sines, dim=1
        )  # sum over k dimension -> (batch_size, seq_len, weather_dim)
        var_p = torch.exp(self.log_var_prior)[:, :seq_len, :].expand(batch_size, -1, -1)

        return mu_p, var_p

    def forward(
        self,
        weather: torch.Tensor,
        coords: torch.Tensor,
        year: torch.Tensor,
        interval: torch.Tensor,
        weather_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        weather: batch_size x seq_len x n_features
        coords: batch_size x 2 (lat, lon) UNNORMALIZED
        year: batch_size x seq_len (UNNORMALIZED, time-varying years)
        interval: batch_size x 1 (UNNORMALIZED in days)
        weather_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len
        """
        batch_size, seq_len, n_features = weather.shape
        # normalize year, interval, and coords
        year, interval, coords = normalize_year_interval_coords(year, interval, coords)

        # Year is [batch_size, seq_len], add feature dimension to make it [batch_size, seq_len, 1]
        year = year.unsqueeze(2)
        # Expand coords to match sequence length if needed
        coords = coords.unsqueeze(1).expand(batch_size, seq_len, 2)

        # mask weather for the masked dimensions
        weather = weather * (~weather_feature_mask)

        input_tensor = torch.cat([weather, year, coords], dim=2)
        # make sure data fits
        input_tensor = input_tensor[:, : self.max_len, :]
        mu_x, var_x = self.transformer_forward(
            input_tensor, src_key_padding_mask=src_key_padding_mask
        )

        # Clip sigma to prevent numerical instability and overly negative log terms
        var_x = torch.clamp(var_x, min=1e-6, max=1)  # sigma is in [0.001, 1]
        mu_p, var_p = self.sinusoidal_prior_forward(weather, interval)

        return mu_x, var_x, mu_p, var_p
