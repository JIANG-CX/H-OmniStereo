#!/usr/bin/env python3
"""Convert raw MVS-GI equidistant fisheye data to the ERP evaluation layout."""

from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from homnistereo.data.loaders.common import decode_float32_rgba  # noqa: E402
from homnistereo.preprocessing import (  # noqa: E402
    copy_file_exclusive,
    create_new_output_root,
    read_split_entries,
    write_image_exclusive,
)

DEFAULT_SPLIT = PROJECT_ROOT / "splits/mvsgi/test_index.txt"
FISHEYE_FOV_DEGREES = 195.0
SOURCE_SIZE = 1024
ERP_HEIGHT = 512
ERP_WIDTH = 1024
CROP_START = 256
CROP_END = 768
MAX_DEPTH_METERS = 128.0
TILT_DEGREES = math.degrees(math.atan2(0.119, 0.28571))
PAIR_CONFIGURATION = {
    "CAM1_0": ("cam1", "cam0", TILT_DEGREES),
    "CAM2_0": ("cam2", "cam0", -TILT_DEGREES),
    "CAM1_2": ("cam1", "cam2", 90.0),
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--max-samples", type=int)
    return parser


def encode_float32_rgba(depth: np.ndarray) -> np.ndarray:
    depth = np.ascontiguousarray(depth, dtype=np.dtype("<f4"))
    if depth.ndim != 2:
        raise ValueError("Depth must have shape [H, W].")
    return depth.view(np.uint8).reshape(depth.shape[0], depth.shape[1], 4)


def _rotation_matrix(roll_degrees: float) -> np.ndarray:
    roll = math.radians(roll_degrees)
    cosine, sine = math.cos(roll), math.sin(roll)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@lru_cache(maxsize=3)
def _erp_remap(roll_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(ERP_WIDTH, dtype=np.float32) + 0.5
    y = np.arange(ERP_HEIGHT, dtype=np.float32) + 0.5
    pixel_x, pixel_y = np.meshgrid(x, y)
    longitude = ((pixel_x - ERP_WIDTH / 2.0) / ERP_WIDTH) * 2.0 * np.pi
    latitude = ((ERP_HEIGHT / 2.0 - pixel_y) / ERP_HEIGHT) * np.pi
    directions = np.stack(
        (
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
            np.cos(latitude) * np.cos(longitude),
        ),
        axis=-1,
    )
    rotated = directions @ _rotation_matrix(roll_degrees).T
    theta = np.arccos(np.clip(rotated[..., 2], -1.0, 1.0))
    alpha = np.arctan2(rotated[..., 1], rotated[..., 0])
    theta_max = math.radians(FISHEYE_FOV_DEGREES) * 0.5
    radius = SOURCE_SIZE * 0.5
    image_radius = radius / theta_max * theta
    valid = (theta <= theta_max) & (image_radius <= radius + 1e-4)
    source_x = SOURCE_SIZE * 0.5 + image_radius * np.cos(alpha)
    source_y = SOURCE_SIZE * 0.5 - image_radius * np.sin(alpha)
    source_x = np.where(valid, source_x, -1.0).astype(np.float32)
    source_y = np.where(valid, source_y, -1.0).astype(np.float32)
    return source_x, source_y, valid


@lru_cache(maxsize=1)
def _fisheye_rays() -> np.ndarray:
    coordinates = np.arange(SOURCE_SIZE, dtype=np.float32) + 0.5
    pixel_x, pixel_y = np.meshgrid(coordinates, coordinates)
    center = SOURCE_SIZE * 0.5
    x = pixel_x - center
    y = pixel_y - center
    radius = np.sqrt(x**2 + y**2)
    theta_max = math.radians(FISHEYE_FOV_DEGREES) * 0.5
    focal_length = np.float32((SOURCE_SIZE * 0.5) / theta_max)
    theta = radius / focal_length
    azimuth = np.arctan2(y, x)
    return np.stack(
        (
            np.sin(theta) * np.cos(azimuth),
            np.sin(theta) * np.sin(azimuth),
            np.cos(theta),
        ),
        axis=-1,
    ).astype(np.float32)


def _read_camera_frame(
    pose_root: Path,
    camera: str,
    frame_name: str,
    uv_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_dir = pose_root / camera
    rgb_path = camera_dir / f"{frame_name}_Fisheye.png"
    depth_path = camera_dir / f"{frame_name}_FisheyeDistance.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    packed_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise FileNotFoundError(f"Unable to read MVS-GI RGB image: {rgb_path}")
    if packed_depth is None:
        raise FileNotFoundError(f"Unable to read MVS-GI depth image: {depth_path}")
    if rgb.shape[:2] != (SOURCE_SIZE, SOURCE_SIZE):
        raise ValueError(f"Unexpected MVS-GI image size at {rgb_path}: {rgb.shape}")
    depth = decode_float32_rgba(packed_depth).astype(np.float32, copy=False)
    valid = (
        (depth > 0)
        & (depth != -1)
        & (depth < MAX_DEPTH_METERS)
        & uv_mask
    )
    depth = depth.copy()
    depth[~valid] = -1.0
    return rgb, depth, valid


def _fisheye_to_erp(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    roll_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    map_x, map_y, geometry_valid = _erp_remap(roll_degrees)
    erp_rgb = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    xyz = _fisheye_rays() * depth[..., None]
    erp_xyz = np.stack(
        [
            cv2.remap(
                xyz[..., channel],
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    erp_mask = cv2.remap(
        valid_mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    erp_mask &= geometry_valid
    erp_depth = np.linalg.norm(erp_xyz, axis=-1).astype(np.float32)
    erp_depth[~erp_mask] = -1.0
    return (
        erp_rgb[:, CROP_START:CROP_END],
        erp_depth[:, CROP_START:CROP_END],
    )


def preprocess_sample(input_root: Path, output_root: Path, entry: Path) -> None:
    if len(entry.parts) < 4:
        raise ValueError(f"Unexpected MVS-GI split entry: {entry}")
    pair, dataset_name, pose_name = entry.parts[:3]
    frame_name = entry.parts[-1]
    if pair not in PAIR_CONFIGURATION:
        raise ValueError(f"Unsupported MVS-GI camera pair in split: {pair}")
    top_camera, bottom_camera, roll_degrees = PAIR_CONFIGURATION[pair]
    pose_root = input_root / dataset_name / dataset_name / pose_name
    uv_mask_path = pose_root / "cam0" / "mask.png"
    uv_mask = cv2.imread(str(uv_mask_path), cv2.IMREAD_GRAYSCALE)
    if uv_mask is None:
        raise FileNotFoundError(f"Unable to read MVS-GI mask: {uv_mask_path}")
    uv_mask = uv_mask > 0

    top_rgb, top_depth, top_valid = _read_camera_frame(
        pose_root, top_camera, frame_name, uv_mask
    )
    bottom_rgb, bottom_depth, bottom_valid = _read_camera_frame(
        pose_root, bottom_camera, frame_name, uv_mask
    )
    top_rgb, top_depth = _fisheye_to_erp(
        top_rgb, top_depth, top_valid, roll_degrees
    )
    bottom_rgb, bottom_depth = _fisheye_to_erp(
        bottom_rgb, bottom_depth, bottom_valid, roll_degrees
    )

    relative_dir = Path(pair) / dataset_name / pose_name
    write_image_exclusive(
        output_root / "UP" / relative_dir / f"{frame_name}_up_rgb.jpg", top_rgb
    )
    write_image_exclusive(
        output_root / "UP" / relative_dir / f"{frame_name}_up_depth.png",
        encode_float32_rgba(top_depth),
    )
    write_image_exclusive(
        output_root / "DOWN" / relative_dir / f"{frame_name}_down_rgb.jpg",
        bottom_rgb,
    )
    write_image_exclusive(
        output_root / "DOWN" / relative_dir / f"{frame_name}_down_depth.png",
        encode_float32_rgba(bottom_depth),
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    input_root = args.input_root.resolve(strict=True)
    entries = read_split_entries(args.split)
    if args.max_samples is not None:
        entries = entries[: args.max_samples]
    output_root = create_new_output_root(input_root, args.output_root)
    copy_file_exclusive(args.split, output_root / args.split.name)
    for entry in tqdm(entries, desc="Preprocessing MVS-GI"):
        preprocess_sample(input_root, output_root, entry)
    print(f"Created {len(entries)} MVS-GI ERP samples in {output_root}")


if __name__ == "__main__":
    main()
