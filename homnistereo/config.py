from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """The single H-OmniStereo architecture released with the paper."""

    corr_radius: int = 4
    corr_levels: int = 2
    n_downsample: int = 2
    n_gru_layers: int = 3
    hidden_dims: tuple[int, int, int] = (128, 128, 128)
    mixed_precision: bool = True
    low_memory: bool = False
    inference_iterations: int = 22
    prior_feature_dim: int = 64
    max_disparity: int = 256
