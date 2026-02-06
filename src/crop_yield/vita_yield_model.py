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

    def yield_model(
        self,
        weather,
        coord,
        year,
        interval,
        weather_feature_mask,
        y_past,
        attention_bias=None,
        return_attention: bool = False,
    ):
        # Apply attention to reduce sequence dimension
        attention_logits = self.weather_attention(weather)  # batch_size x seq_len x 1
        if attention_bias is not None:
            attention_logits = attention_logits + attention_bias
        if weather_feature_mask is not None and attention_logits.shape[:2] == weather_feature_mask.shape[:2]:
            # mask padded steps by setting logits to large negative where all features are masked
            valid_steps = (~weather_feature_mask).any(dim=2, keepdim=True)  # (B, seq, 1)
            attention_logits = attention_logits.masked_fill(~valid_steps, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=1)

        # Apply attention to get weighted sum
        weather_attended = torch.sum(
            weather * attention_weights, dim=1
        )  # batch_size x weather_dim

        mlp_input = torch.cat([weather_attended, y_past], dim=1)
        yield_pred = self.yield_mlp(mlp_input)
        if return_attention:
            return yield_pred, attention_weights
        return yield_pred

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
        # Handle dict checkpoints
        if isinstance(pretrained_model, dict):
            if "model_state_dict" in pretrained_model:
                state_dict = pretrained_model["model_state_dict"]
            elif "state_dict" in pretrained_model:
                state_dict = pretrained_model["state_dict"]
            else:
                state_dict = pretrained_model
            if "positional_encoding.pos_encoding" in state_dict:
                pos_encoding = state_dict["positional_encoding.pos_encoding"]
                state_dict["positional_encoding.pos_encoding"] = pos_encoding[
                    : self.weather_model.max_len, :
                ]

            self.weather_model.load_state_dict(state_dict, strict=False)
            return

        # Handle model objects
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
        attention_bias=None,
        return_attention: bool = False,
    ):
        """
        weather: batch_size x seq_len x n_features
        coord: batch_size x 2 (lat, lon) UNNORMALIZED
        year: batch_size x seq_len (UNNORMALIZED, time-varying years)
        interval: batch_size x 1 (UNNORMALIZED in days)
        weather_feature_mask: batch_size x seq_len x n_features
        y_past: batch_size x n_past_years+1
        attention_bias: optional tensor (batch_size x seq_len x 1) added to attention logits
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
        yield_outputs = self.yield_model(
            z_imputed,
            coord,
            year,
            interval,
            weather_feature_mask=weather_feature_mask,
            y_past=y_past,
            attention_bias=attention_bias,
            return_attention=return_attention,
        )
        if return_attention:
            yield_pred, attention_weights = yield_outputs
            return yield_pred, z, mu_x, var_x, mu_p, var_p, attention_weights
        return yield_outputs, z, mu_x, var_x, mu_p, var_p
