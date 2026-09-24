from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawSample:
    """One dataset sample before common resize and tensor conversion."""

    top_image: np.ndarray
    bottom_image: np.ndarray
    disparity: np.ndarray
    valid_mask: np.ndarray
    depth: np.ndarray
    baseline: float
    fov_params: tuple[float, float, float, float] | list[float]
