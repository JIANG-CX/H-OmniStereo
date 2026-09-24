from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from homnistereo.data.loaders.common import read_rgb
from homnistereo.data.types import RawSample
from homnistereo.geometry import depth_uint8_decoding, disp_to_depth

MAX_DEPTH_METERS = 128.0
FOV_PARAMS = (0, 360, 0, 180)


def load_sample(
    dataset_root: Path,
    relative_path: str,
    sample_name: str,
) -> RawSample:
    sample_dir = dataset_root / relative_path
    top_path = sample_dir / f"{sample_name}_up.jpg"
    bottom_path = sample_dir / f"{sample_name}_down.jpg"
    disparity_path = sample_dir / f"{sample_name}_downdisp.png"
    config_path = sample_dir / f"{sample_name}_camera_params.json"

    top_image = read_rgb(top_path)
    bottom_image = read_rgb(bottom_path)
    disparity = depth_uint8_decoding(imageio.imread(disparity_path), scale=4096)
    with config_path.open(encoding="utf-8") as stream:
        baseline = float(json.load(stream)["baseline"])
    max_disparity = top_image.shape[0] // 2
    valid_mask = (disparity > 1e-7) & (disparity < max_disparity)
    depth = disp_to_depth(
        disparity,
        baseline=np.asarray([baseline]),
        valid_mask=valid_mask,
        fov_params=np.asarray(FOV_PARAMS),
    )
    valid_mask &= (depth > 0) & (depth < MAX_DEPTH_METERS)
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
