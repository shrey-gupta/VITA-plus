import torch
import torch.nn as nn
from src.base.vita import VITA
from src.base.base_model import BaseModel


class VITAYieldModel(BaseModel):
    """
    VITA-based yield prediction model that handles probabilistic weather representations
    with sinusoidal priors.
    """

    def __init__(
        self,
        name: str,
        device: torch.device,
        k: int,
        weather_dim: int,
        n_past_years: int,
        **model_size_params,
    ):
        super().__init__(name)
        self.weather_model = VITA(
            weather_dim=weather_dim,
            output_dim=weather_dim,
            k=k,
            device=device,
            **model_size_params,
        )

        # Attention mechanism to reduce sequence dimension
        self.weather_attention = nn.Sequential(
            nn.Linear(weather_dim, 16), nn.GELU(), nn.Linear(16, 1)
        )

        self.yield_mlp = nn.Sequential(
            nn.Linear(weather_dim + n_past_years + 1, 120),
            nn.GELU(),
            nn.Linear(120, 1),
        )

        self.weather_model_frozen = False

    def yield_model(self, weather, coord, year, interval, weather_feature_mask, y_past):
        # Apply attention to reduce sequence dimension
        attention_weights = self.weather_attention(weather)  # batch_size x seq_len x 1
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention to get weighted sum
        weather_attended = torch.sum(
            weather * attention_weights, dim=1
        )  # batch_size x weather_dim

        mlp_input = torch.cat([weather_attended, y_past], dim=1)
        return self.yield_mlp(mlp_input)

    def _impute_weather(self, original_weather, imputed_weather, weather_feature_mask):
        """
        Fast combination using element-wise ops instead of torch.where:
        - original_weather: batch_size x seq_len x weather_dim
        - imputed_weather: batch_size x seq_len x weather_dim
        - weather_feature_mask: batch_size x seq_len x weather_dim
        """
        return (
            original_weather * (~weather_feature_mask)
            + imputed_weather * weather_feature_mask
        )

    def load_pretrained(self, pretrained_model):
        """Load pretrained weather model weights"""
        self.logger.info(f"provided model class: {pretrained_model.__class__.__name__}")
        if isinstance(pretrained_model, VITA):
            weather_model = pretrained_model
        elif isinstance(pretrained_model, VITAYieldModel):
            weather_model = pretrained_model.weather_model
            self.weather_attention = pretrained_model.weather_attention
            self.yield_mlp = pretrained_model.yield_mlp
        else:
            raise ValueError(
                f"provided model class: {pretrained_model.__class__.__name__} is not supported"
            )

        self.weather_model.load_pretrained(weather_model)

    def forward(
        self,
        weather,
        coord,
        year,
        interval,
        weather_feature_mask,
        y_past,
        src_key_padding_mask=None,
    ):
        """
        weather: batch_size x seq_len x n_features
        coord: batch_size x 2 (lat, lon) UNNORMALIZED
        year: batch_size x seq_len (UNNORMALIZED, time-varying years)
        interval: batch_size x 1 (UNNORMALIZED in days)
        weather_feature_mask: batch_size x seq_len x n_features
        y_past: batch_size x n_past_years+1
        """
        # VITA returns (mu_x, var_x, mu_p, var_p)
        mu_x, var_x, mu_p, var_p = self.weather_model(
            weather,
            coord,
            year,
            interval,
            weather_feature_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Apply reparameterization trick: z = mu + sigma * epsilon
        epsilon = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * epsilon

        # use vita predicted weather only for masked features
        z_imputed = self._impute_weather(weather, z, weather_feature_mask)

        # Predict yield using imputed weather
        yield_pred = self.yield_model(
            z_imputed,
            coord,
            year,
            interval,
            weather_feature_mask=None,
            y_past=y_past,
        )
        return yield_pred, z, mu_x, var_x, mu_p, var_p
