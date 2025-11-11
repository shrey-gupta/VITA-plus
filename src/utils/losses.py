"""
Custom loss functions.
All loss functions return output of shape [batch_size]
"""

import torch
from typing import Tuple, Optional


def gaussian_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    feature_mask: torch.Tensor,
    masked_dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Compute the Gaussian log-likelihood for masked features.
    Log-likelihood = -0.5 * log(2πσ²) - (x-μ)²/(2σ²)
    """
    if masked_dims is None:
        # Default to summing over all dimensions except batch dimension
        masked_dims = tuple(range(1, x.ndim))
    # Compute the Gaussian log-likelihood
    log_likelihood = -0.5 * torch.log(2 * torch.pi * var) - 0.5 * (x - mu) ** 2 / var
    masked_log_likelihood = log_likelihood * feature_mask
    return torch.sum(masked_log_likelihood, dim=masked_dims)


def compute_gaussian_kl_divergence(
    feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features]
    mu_x: torch.Tensor,  # [batch_size, seq_len, n_features]
    var_x: torch.Tensor,  # [batch_size, seq_len, n_features]
    mu_p: torch.Tensor,  # [1, n_mixtures, n_features]
    var_p: torch.Tensor,  # [1, n_mixtures, n_features]
) -> torch.Tensor:
    """
    Compute KL divergence between two diagonal Gaussians for masked features only.
    KL(q(z|x) || p(z)) = 0.5 * [log(var_p/var_x) + var_x/var_p + (mu_x - mu_p)^2/var_p - 1]
    """
    kl_per_dim = 0.5 * (
        torch.log(var_p / var_x) + var_x / var_p + (mu_x - mu_p) ** 2 / var_p - 1.0
    )
    kl_masked = kl_per_dim * feature_mask
    # sum over masked dims
    kl_divergence = torch.sum(kl_masked, dim=(1, 2))
    return kl_divergence
