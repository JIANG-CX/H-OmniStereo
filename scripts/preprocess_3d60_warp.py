#!/usr/bin/env python3
"""Create the rotated 3D60-Warp evaluation set from the original 3D60 data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from homnistereo.preprocessing import (  # noqa: E402
    copy_file_exclusive,
    create_new_output_root,
    read_split_entries,
    rotate_erp_batch,
    write_image_exclusive,
)

DEFAULT_SPLIT = PROJECT_ROOT / "splits/threeD60warp/test_index.txt"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    return parser


def _read_required(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Unable to read required 3D60 file: {path}")
    return image


def _source_paths(input_root: Path, entry: Path) -> dict[str, Path]:
    relative_dir = entry.parent
    sample_name = entry.name
    return {
        "right_rgb": input_root
        / "Right"
        / relative_dir
        / f"{sample_name}_color_0_Right_0.0.png",
        "right_depth": input_root
        / "Right"
        / relative_dir
        / f"{sample_name}_depth_0_Right_0.0.exr",
        "up_rgb": input_root
        / "Up"
        / relative_dir
        / f"{sample_name}_color_0_Up_0.0.png",
        "up_depth": input_root
        / "Up"
        / relative_dir
        / f"{sample_name}_depth_0_Up_0.0.exr",
    }


def preprocess_sample(
    input_root: Path,
    output_root: Path,
    entry: Path,
    device: torch.device,
) -> None:
    source = _source_paths(input_root, entry)
    right_rgb = _read_required(source["right_rgb"], cv2.IMREAD_COLOR)
    up_rgb = _read_required(source["up_rgb"], cv2.IMREAD_COLOR)
    right_depth = _read_required(source["right_depth"], cv2.IMREAD_ANYDEPTH)
    up_depth = _read_required(source["up_depth"], cv2.IMREAD_ANYDEPTH)
    if right_rgb.shape != up_rgb.shape or right_depth.shape != up_depth.shape:
        raise ValueError(f"Mismatched 3D60 pair dimensions for {entry}")

    rgb_tensor = torch.from_numpy(np.stack((right_rgb, up_rgb))).to(device)
    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2).float()
    rotated_rgb = rotate_erp_batch(rgb_tensor, roll_degrees=45.0)
    rotated_rgb = rotated_rgb.clamp(0, 255).to(torch.uint8)
    rotated_rgb = rotated_rgb.permute(0, 2, 3, 1).cpu().numpy()

    depth_tensor = torch.from_numpy(
        np.stack((right_depth, up_depth)).astype(np.float32, copy=False)
    ).to(device)
    rotated_depth = rotate_erp_batch(
        depth_tensor[:, None], roll_degrees=45.0
    )[:, 0].cpu().numpy()

    relative_dir = entry.parent
    sample_name = entry.name
    right_dir = output_root / "Right_warp" / relative_dir
    up_dir = output_root / "Up_warp" / relative_dir
    write_image_exclusive(
        right_dir / f"{sample_name}_color_0_Right_warp_0.0.png",
        rotated_rgb[0],
    )
    write_image_exclusive(
        right_dir / f"{sample_name}_depth_0_Right_warp_0.0.exr",
        rotated_depth[0],
    )
    write_image_exclusive(
        up_dir / f"{sample_name}_color_0_Up_warp_0.0.png",
        rotated_rgb[1],
    )
    write_image_exclusive(
        up_dir / f"{sample_name}_depth_0_Up_warp_0.0.exr",
        rotated_depth[1],
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    input_root = args.input_root.resolve(strict=True)
    entries = read_split_entries(args.split)
    if args.max_samples is not None:
        entries = entries[: args.max_samples]
    output_root = create_new_output_root(input_root, args.output_root)
    copy_file_exclusive(args.split, output_root / args.split.name)
    device = torch.device(args.device)
    for entry in tqdm(entries, desc="Preprocessing 3D60-Warp"):
        preprocess_sample(input_root, output_root, entry, device)
    print(f"Created {len(entries)} 3D60-Warp samples in {output_root}")


if __name__ == "__main__":
    main()
