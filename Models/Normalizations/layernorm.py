import torch
import torch.nn as nn
from typing import Union, List


class LayerNorm(nn.Module):
    """
    Fixed LayerNorm: Normalises over the channel dimension (C) independently 
    for each position in the sequence, preserving word semantics

    y = (x - mean) / sqrt(var + eps) * weight + bias
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
    ):
        super().__init__()
        # if isinstance(normalized_shape, int):
        #     normalized_shape = [normalized_shape]
        # self.normalized_shape = list(normalized_shape)
        ### changed - extract C and ignore L
        self.C = normalized_shape[0] if isinstance(normalized_shape, list) else normalized_shape
        self.eps = eps

        # Learnable affine parameters (same shape as normalized_shape)
        ### changed - to per channel 
        self.weight = nn.Parameter(torch.ones(1, self.C, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.C, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine which dims to reduce over (the last N dims)
        # n = len(self.normalized_shape)
        # dims = tuple(range(-n, 0))  # e.g. (-2, -1) for a 2-D normalized_shape

        ### changed - dim=1
        #mean = x.mean(dim=dims, keepdim=False)
        mean = x.mean(dim=1, keepdim=True)
        #var = x.var(dim=dims, keepdim=False, unbiased=False)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias