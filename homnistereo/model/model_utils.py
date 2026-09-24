"""Small tensor samplers used by stereo correlation volumes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bilinear_sampler(image, coordinates, mode="bilinear", mask=False, low_memory=False):
    """Sample a one-dimensional stereo volume using pixel coordinates."""
    height, width = image.shape[-2:]
    x_grid, y_grid = coordinates.split([1, 1], dim=-1)
    x_grid = 2 * x_grid / (width - 1) - 1
    if torch.unique(y_grid).numel() != 1 or height != 1:
        raise ValueError("Stereo volume sampling expects a singleton y dimension.")
    grid = torch.cat([x_grid, y_grid], dim=-1).to(image.dtype)
    with torch.backends.cudnn.flags(enabled=False):
        sampled = F.grid_sample(image, grid, mode=mode, align_corners=True)
    if mask:
        valid = (x_grid > -1) & (y_grid > -1) & (x_grid < 1) & (y_grid < 1)
        return sampled, valid.float()
    return sampled
