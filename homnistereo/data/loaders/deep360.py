from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from homnistereo.data.loaders.common import read_rgb
from homnistereo.data.types import RawSample
from homnistereo.geometry import disp_to_depth

BASELINES_METERS = {
    "12": 1.0,
    "13": 1.0,
    "14": math.sqrt(2.0),
    "23": math.sqrt(2.0),
    "24": 1.0,
    "34": 1.0,
}
PAIR_CAMERAS = {
    "12": ("1", "2"),
    "13": ("1", "3"),
    "14": ("1", "4"),
    "23": ("2", "3"),
    "24": ("2", "4"),
    "34": ("3", "4"),
}
MAX_DEPTH_METERS = 128.0
MAX_DISPARITY_PIXELS = 192.0
FOV_PARAMS = (0, 360, 0, 180)


def load_sample(
    dataset_root: Path,
    relative_path: str,
    sample_name: str,
) -> RawSample:
    stereo_pair = sample_name.rsplit("_", maxsplit=1)[-1]
    if stereo_pair not in BASELINES_METERS:
        supported = ", ".join(BASELINES_METERS)
        raise ValueError(
            f"Unknown Deep360 stereo pair {stereo_pair!r}; expected one of {supported}."
        )
    left_camera, right_camera = PAIR_CAMERAS[stereo_pair]
    sample_dir = dataset_root / relative_path
    left_path = sample_dir / "rgb" / f"{sample_name}_rgb{left_camera}.png"
    right_path = sample_dir / "rgb" / f"{sample_name}_rgb{right_camera}.png"
    disparity_path = sample_dir / "disp" / f"{sample_name}_disp.npz"

    bottom_image = cv2.rotate(read_rgb(left_path), cv2.ROTATE_90_COUNTERCLOCKWISE)
    top_image = cv2.rotate(read_rgb(right_path), cv2.ROTATE_90_COUNTERCLOCKWISE)
    with np.load(disparity_path) as payload:
        horizontal_disparity = payload["arr_0"].astype(np.float32)
    disparity = cv2.rotate(horizontal_disparity, cv2.ROTATE_90_COUNTERCLOCKWISE)
    valid_mask = (
        (disparity > 1e-7)
        & np.isfinite(disparity)
        & (disparity <= MAX_DISPARITY_PIXELS)
    )
    baseline = BASELINES_METERS[stereo_pair]
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
