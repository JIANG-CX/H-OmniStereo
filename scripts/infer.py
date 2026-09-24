#!/usr/bin/env python3
"""Run H-OmniStereo on an arbitrary top/bottom ERP image pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from homnistereo.predictor import HOmniStereoPredictor  # noqa: E402
from homnistereo.visualization import (  # noqa: E402
    save_point_cloud_ply,
    save_prediction_visuals,
)

DEFAULT_EXAMPLE_DIR = PROJECT_ROOT / "assets/360sdnet/stairs"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=Path, default=DEFAULT_EXAMPLE_DIR / "top.png")
    parser.add_argument(
        "--bottom", type=Path, default=DEFAULT_EXAMPLE_DIR / "bottom.png"
    )
    parser.add_argument(
        "--camera-params",
        type=Path,
        default=DEFAULT_EXAMPLE_DIR / "camera.json",
        help="JSON file containing baseline and ERP field-of-view parameters.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/h_omnistereo.pth"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/inference/stairs"
    )
    parser.add_argument("--iterations", type=int, default=22)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--point-cloud", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_argument_parser().parse_args()


def load_camera_parameters(path: Path) -> tuple[float, tuple[float, float, float, float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid camera parameter JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Camera parameter file must contain a JSON object.")
    if "baseline" not in payload or "fov_params" not in payload:
        raise ValueError(
            "Camera parameter file must define 'baseline' and 'fov_params'."
        )
    try:
        baseline = float(payload["baseline"])
        fov_params = tuple(float(value) for value in payload["fov_params"])
    except (TypeError, ValueError) as error:
        raise ValueError("Camera parameters must be numeric.") from error
    if len(fov_params) != 4:
        raise ValueError("'fov_params' must contain exactly four values.")
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("Camera baseline must be a positive finite value.")
    if not np.all(np.isfinite(fov_params)):
        raise ValueError("ERP field-of-view parameters must be finite.")
    if fov_params[1] <= fov_params[0] or fov_params[3] <= fov_params[2]:
        raise ValueError("ERP longitude and latitude ranges must be increasing.")
    return baseline, fov_params


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _network_resolution(height: int, width: int) -> tuple[int, int]:
    return int(np.ceil(height / 32) * 32), int(np.ceil(width / 32) * 32)


def main() -> None:
    args = parse_args()
    baseline, fov_params = load_camera_parameters(args.camera_params)

    top_original = load_rgb(args.top)
    bottom_original = load_rgb(args.bottom)
    if top_original.shape != bottom_original.shape:
        raise ValueError("Top and bottom images must have identical dimensions.")
    original_height, original_width = top_original.shape[:2]
    height, width = _network_resolution(original_height, original_width)
    if (height, width) == (original_height, original_width):
        top, bottom = top_original, bottom_original
    else:
        top = cv2.resize(top_original, (width, height), interpolation=cv2.INTER_LINEAR)
        bottom = cv2.resize(
            bottom_original, (width, height), interpolation=cv2.INTER_LINEAR
        )

    predictor = HOmniStereoPredictor(args.checkpoint, args.device, args.iterations)
    prediction = predictor.predict(
        torch.from_numpy(top).permute(2, 0, 1).unsqueeze(0),
        torch.from_numpy(bottom).permute(2, 0, 1).unsqueeze(0),
        torch.tensor([baseline]),
        torch.tensor([fov_params]),
    )
    disparity = prediction.disparity[0].cpu().numpy()
    depth = prediction.depth[0].cpu().numpy()
    uncertainty = prediction.uncertainty[0].cpu().numpy()
    valid_mask = prediction.valid_mask[0].cpu().numpy()
    save_prediction_visuals(
        args.output_dir,
        disparity,
        depth,
        uncertainty,
        valid_mask,
    )
    cv2.imwrite(str(args.output_dir / "top.png"), cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(args.output_dir / "bottom.png"), cv2.cvtColor(bottom, cv2.COLOR_RGB2BGR)
    )
    if args.point_cloud:
        save_point_cloud_ply(
            args.output_dir / "point_cloud.ply",
            depth,
            bottom,
            valid_mask,
            np.asarray(fov_params),
        )
    metadata = {
        "architecture": "H-OmniStereo",
        "top_image": str(args.top),
        "bottom_image": str(args.bottom),
        "original_resolution": [original_width, original_height],
        "inference_resolution": [width, height],
        "camera_parameters": str(args.camera_params),
        "baseline_m": baseline,
        "fov_params": list(fov_params),
        "checkpoint": str(args.checkpoint),
        "iterations": args.iterations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved inference outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
