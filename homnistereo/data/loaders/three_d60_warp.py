from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from homnistereo.data.loaders.common import read_any_depth, read_rgb
from homnistereo.data.types import RawSample
from homnistereo.geometry import depth_to_disp_pix

BASELINE_METERS = math.sqrt(0.2834**2 + 0.2834**2)
MAX_DEPTH_METERS = 1000.0
FOV_PARAMS = (0.0, 360.0, 0.0, 180.0)


def load_sample(
    dataset_root: Path,
    relative_path: str,
    sample_name: str,
) -> RawSample:
    sample_dir = Path(relative_path)
    top_path = (
        dataset_root / "Up_warp" / sample_dir / f"{sample_name}_color_0_Up_warp_0.0.png"
    )
    bottom_dir = dataset_root / "Right_warp" / sample_dir
    bottom_path = bottom_dir / f"{sample_name}_color_0_Right_warp_0.0.png"
    depth_path = bottom_dir / f"{sample_name}_depth_0_Right_warp_0.0.exr"

    top_image = read_rgb(top_path)
    bottom_image = read_rgb(bottom_path)
    depth = read_any_depth(depth_path)
    disparity = depth_to_disp_pix(
        depth, baseline=BASELINE_METERS, fov_params=np.asarray(FOV_PARAMS)
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
        baseline=BASELINE_METERS,
        fov_params=FOV_PARAMS,
    )
