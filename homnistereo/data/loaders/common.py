from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_unchanged(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def read_any_depth(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(f"Unable to read depth image: {path}")
    return image


def decode_float32_rgba(encoded: np.ndarray) -> np.ndarray:
    """Decode little-endian float32 values packed into four uint8 channels."""
    if encoded.ndim != 3 or encoded.shape[2] != 4:
        raise ValueError(f"Expected an HxWx4 encoded depth image, got {encoded.shape}.")
    return np.ascontiguousarray(encoded).view("<f4").squeeze(-1)
