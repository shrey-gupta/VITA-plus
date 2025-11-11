import math
import torch
import torch.nn as nn

"""
This class implements the classic sinusoidal positional encoding introduced in the
"Attention is All You Need" paper.
"""


class VanillaPositionalEncoding(nn.Module):
    pos_encoding: torch.Tensor  # Type hint for the buffer

    def __init__(self, hidden_dim, max_len, device):
        assert (
            hidden_dim % 2 == 0
        ), "hidden_dim should be divisible by 2 for separate encoding"

        super(VanillaPositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim

        # Pre-compute positional encodings
        pos_encoding = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create frequency terms
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        # Apply sine to even indices
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (will be moved to device with model)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward method for adding positional encoding.

        Args:
        token_embedding: Tensor, shape [batch_size, seq_len, d_model]

        Returns:
        Tensor with positional encoding added, same shape as x.
        """
        batch_size, seq_len, hidden_dim = token_embedding.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: got {hidden_dim} != expected {self.hidden_dim}"
            )

        # Add positional encoding to input
        token_embedding = token_embedding + self.pos_encoding[:seq_len, :].unsqueeze(0)
        return token_embedding
