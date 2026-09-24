from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from homnistereo.data.loaders.common import (
    decode_float32_rgba,
    read_rgb,
    read_unchanged,
)
from homnistereo.data.types import RawSample
from homnistereo.geometry import depth_to_disp_pix

MAX_DEPTH_METERS = 128.0
FOV_PARAMS = (90.0, 270.0, 0.0, 180.0)


def _baseline_for(relative_path: str) -> float:
    if "CAM1_0" in relative_path or "CAM2_0" in relative_path:
        return math.sqrt(0.119**2 + 0.28571**2)
    if "CAM1_2" in relative_path:
        return 0.238
    raise ValueError(f"Cannot infer the MVS-GI baseline from {relative_path!r}.")


def load_sample(
    dataset_root: Path,
    relative_path: str,
    sample_name: str,
) -> RawSample:
    sample_dir = Path(relative_path)
    top_path = dataset_root / "UP" / sample_dir / f"{sample_name}_up_rgb.jpg"
    bottom_dir = dataset_root / "DOWN" / sample_dir
    bottom_path = bottom_dir / f"{sample_name}_down_rgb.jpg"
    depth_path = bottom_dir / f"{sample_name}_down_depth.png"
    baseline = _baseline_for(relative_path)

    top_image = read_rgb(top_path)
    bottom_image = read_rgb(bottom_path)
    depth = decode_float32_rgba(read_unchanged(depth_path)).astype(
        np.float32, copy=False
    )
    disparity = depth_to_disp_pix(
        depth, baseline=baseline, fov_params=np.asarray(FOV_PARAMS)
    )
    valid_mask = (depth > 0) & (depth < MAX_DEPTH_METERS) & (disparity > 1e-7)

    disparity[~valid_mask] = 0
    depth[~valid_mask] = 0
    return RawSample(
        top_image=top_image,
        bottom_image=bottom_image,
        disparity=disparity,
        valid_mask=valid_mask,
        depth=depth,
        baseline=baseline,
        fov_params=FOV_PARAMS,
    )
