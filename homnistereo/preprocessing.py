"""Shared, overwrite-safe helpers for public dataset preprocessing."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def read_split_entries(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {path}")

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = Path(line)
        if entry.is_absolute() or ".." in entry.parts:
            raise ValueError(f"Split entries must be relative paths: {line}")
        entries.append(entry)
    if not entries:
        raise ValueError(f"Split file is empty: {path}")
    return entries


def create_new_output_root(input_root: str | Path, output_root: str | Path) -> Path:
    input_root = Path(input_root).resolve(strict=True)
    output_root = Path(output_root).resolve(strict=False)
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("Output root must be separate from and outside the input root.")
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists; choose a new path to avoid overwriting data: "
            f"{output_root}"
        )
    output_root.mkdir(parents=True)
    return output_root


def copy_file_exclusive(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while chunk := input_stream.read(1024 * 1024):
            output_stream.write(chunk)


def write_image_exclusive(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    encode_parameters = []
    if extension in {".jpg", ".jpeg"}:
        encode_parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
    success, encoded = cv2.imencode(extension, image, encode_parameters)
    if not success:
        raise OSError(f"OpenCV could not encode output image: {path}")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def euler_rotation_matrix(
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    angles = torch.deg2rad(
        torch.tensor(
            [yaw_degrees, pitch_degrees, roll_degrees],
            device=device,
            dtype=torch.float32,
        )
    )
    yaw, pitch, roll = angles.unbind()
    zero = torch.zeros((), device=device)
    one = torch.ones((), device=device)
    rotation_z = torch.stack(
        (
            torch.stack((torch.cos(yaw), -torch.sin(yaw), zero)),
            torch.stack((torch.sin(yaw), torch.cos(yaw), zero)),
            torch.stack((zero, zero, one)),
        )
    )
    rotation_y = torch.stack(
        (
            torch.stack((torch.cos(pitch), zero, torch.sin(pitch))),
            torch.stack((zero, one, zero)),
            torch.stack((-torch.sin(pitch), zero, torch.cos(pitch))),
        )
    )
    rotation_x = torch.stack(
        (
            torch.stack((one, zero, zero)),
            torch.stack((zero, torch.cos(roll), -torch.sin(roll))),
            torch.stack((zero, torch.sin(roll), torch.cos(roll))),
        )
    )
    return rotation_z @ rotation_y @ rotation_x


def rotate_erp_batch(
    images: torch.Tensor,
    *,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    roll_degrees: float = 0.0,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Rotate a BCHW ERP tensor using the paper preprocessing convention."""
    if images.ndim != 4:
        raise ValueError("ERP input must have shape [B, C, H, W].")
    if mode not in {"bilinear", "nearest"}:
        raise ValueError("Interpolation mode must be 'bilinear' or 'nearest'.")

    batch_size, _, height, width = images.shape
    device = images.device
    rotation = euler_rotation_matrix(
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
        device=device,
    )
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    longitude = x[None, :] * (2.0 * torch.pi) - torch.pi
    latitude = (0.5 - y[:, None]) * torch.pi
    directions = torch.stack(
        (
            torch.cos(latitude) * torch.cos(longitude),
            torch.sin(latitude).expand(height, width),
            torch.cos(latitude) * torch.sin(longitude),
        ),
        dim=-1,
    ).reshape(height * width, 3)
    input_directions = directions @ rotation
    input_latitude = torch.asin(input_directions[:, 1].clamp(-1.0, 1.0))
    input_longitude = torch.atan2(input_directions[:, 2], input_directions[:, 0])
    source_x = torch.remainder(
        (input_longitude + torch.pi) / (2.0 * torch.pi) * width - 0.5 + 1e-7,
        width,
    )
    source_y = (
        (0.5 - input_latitude / torch.pi) * height - 0.5
    ).clamp(0.0, height - 1.0 - 1e-7)
    grid = torch.stack(
        (
            source_x / (width - 1.0) * 2.0 - 1.0,
            source_y / (height - 1.0) * 2.0 - 1.0,
        ),
        dim=-1,
    ).reshape(1, height, width, 2)
    return F.grid_sample(
        images.float(),
        grid.expand(batch_size, -1, -1, -1),
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
