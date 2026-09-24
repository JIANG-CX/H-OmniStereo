from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from homnistereo.geometry import compute_padding_params, padding_resize_image

from .loaders.deep360 import load_sample as load_deep360_sample
from .loaders.generated_omnidirectional import load_sample as load_generated_sample
from .loaders.mvsgi import load_sample as load_mvsgi_sample
from .loaders.three_d60 import load_sample as load_three_d60_sample
from .loaders.three_d60_warp import load_sample as load_three_d60_warp_sample
from .types import RawSample

DATASET_LOADERS = {
    "threeD60": load_three_d60_sample,
    "threeD60warp": load_three_d60_warp_sample,
    "mvsgi": load_mvsgi_sample,
    "ours": load_generated_sample,
    "deep360": load_deep360_sample,
}


def load_dataset_registry(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    for specification in registry["datasets"].values():
        root = Path(specification["root"])
        if not root.is_absolute():
            specification["root"] = str((path.parent.parent / root).resolve())
    return registry


def _read_index(index_path: Path) -> list[tuple[str, str]]:
    if not index_path.is_file():
        raise FileNotFoundError(f"Dataset split file does not exist: {index_path}")
    entries = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = Path(line)
        entries.append(
            (entry.name, "." if entry.parent == Path(".") else str(entry.parent))
        )
    if not entries:
        raise ValueError(f"Dataset split file is empty: {index_path}")
    return entries


def _resize_array(array, width: int, height: int, pad_params: dict, interpolation: int):
    if array is None:
        return None
    return padding_resize_image(
        array,
        width,
        height,
        padding_mode=None,
        pad_params=pad_params,
        interpolation=interpolation,
    )


def _prepare_sample(
    sample: RawSample,
    specification: dict[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    width, height = specification["image_size"]
    height = int(np.ceil(height / 32) * 32)
    width = (
        height * 2 if specification["force_2_to_1"] else int(np.ceil(width / 32) * 32)
    )

    raw_height, raw_width = sample.top_image.shape[:2]
    lon_min, lon_max, lat_min, lat_max = sample.fov_params
    pad_params = compute_padding_params(
        raw_height,
        raw_width,
        width,
        height,
        lat_degree_range=[lat_min, lat_max],
        lon_degree_range=[lon_min, lon_max],
    )
    disparity_scale = width / pad_params["padded_w"]
    fov_params = [
        *pad_params["new_lon_degree_range"],
        *pad_params["new_lat_degree_range"],
    ]

    top_image = _resize_array(
        sample.top_image, width, height, pad_params, cv2.INTER_LINEAR
    )
    bottom_image = _resize_array(
        sample.bottom_image, width, height, pad_params, cv2.INTER_LINEAR
    )
    disparity = (
        _resize_array(sample.disparity, width, height, pad_params, cv2.INTER_NEAREST)
        * disparity_scale
    )
    valid_mask = _resize_array(
        sample.valid_mask.astype(np.uint8),
        width,
        height,
        pad_params,
        cv2.INTER_NEAREST,
    ).astype(bool)
    depth = _resize_array(sample.depth, width, height, pad_params, cv2.INTER_NEAREST)

    # Vertical disparity is undefined at the two ERP poles.
    valid_mask[0] = False
    valid_mask[-1] = False

    return {
        "top_image": torch.from_numpy(np.ascontiguousarray(top_image)).permute(2, 0, 1),
        "bottom_image": torch.from_numpy(np.ascontiguousarray(bottom_image)).permute(
            2, 0, 1
        ),
        "disparity": torch.from_numpy(np.asarray(disparity, dtype=np.float32)),
        "depth": torch.from_numpy(np.asarray(depth, dtype=np.float32)),
        "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask)),
        "baseline": torch.tensor(sample.baseline, dtype=torch.float32),
        "fov_params": torch.tensor(fov_params, dtype=torch.float32),
        "sample_id": sample_id,
    }


class EvaluationDataset(Dataset):
    def __init__(self, name: str, specification: dict[str, Any]) -> None:
        if name not in DATASET_LOADERS:
            raise ValueError(f"No loader registered for dataset {name!r}.")
        self.name = name
        self.specification = specification
        self.root = Path(specification["root"])
        self.entries = _read_index(self.root / specification["split"])
        self.loader = DATASET_LOADERS[name]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename, relative_path = self.entries[index]
        raw_sample = self.loader(self.root, relative_path, filename)
        sample_id = str(Path(relative_path) / filename)
        return _prepare_sample(raw_sample, self.specification, sample_id)


def evaluation_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return default_collate(samples)
