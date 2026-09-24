from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from homnistereo.geometry import depth_to_points


def colorize_scalar(
    values: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).squeeze()
    valid = np.isfinite(values)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(values[valid], (2.0, 98.0))
        if high <= low:
            high = low + 1e-6
        normalized[valid] = np.clip(
            (values[valid] - low) / (high - low) * 255.0, 0, 255
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def save_prediction_visuals(
    output_dir: str | Path,
    disparity: np.ndarray,
    depth: np.ndarray,
    uncertainty: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "disparity.npy", disparity)
    np.save(output_dir / "depth.npy", depth)
    np.save(output_dir / "uncertainty.npy", uncertainty)
    cv2.imwrite(
        str(output_dir / "disparity.png"), colorize_scalar(disparity, valid_mask)
    )
    cv2.imwrite(str(output_dir / "depth.png"), colorize_scalar(depth, valid_mask))
    cv2.imwrite(
        str(output_dir / "uncertainty.png"), colorize_scalar(uncertainty, valid_mask)
    )


def save_evaluation_visuals(
    output_dir: str | Path,
    prediction: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> None:
    """Save prediction, target, and absolute-error views for one sample."""
    output_dir = Path(output_dir)
    save_prediction_visuals(
        output_dir / "prediction",
        prediction["disparity"],
        prediction["depth"],
        prediction["uncertainty"],
        valid_mask,
    )
    target_dir = output_dir / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "disparity.npy", target["disparity"])
    np.save(target_dir / "depth.npy", target["depth"])
    cv2.imwrite(
        str(target_dir / "disparity.png"),
        colorize_scalar(target["disparity"], valid_mask),
    )
    cv2.imwrite(
        str(target_dir / "depth.png"), colorize_scalar(target["depth"], valid_mask)
    )
    error_dir = output_dir / "absolute_error"
    error_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(error_dir / "disparity.png"),
        colorize_scalar(
            np.abs(prediction["disparity"] - target["disparity"]), valid_mask
        ),
    )
    cv2.imwrite(
        str(error_dir / "depth.png"),
        colorize_scalar(np.abs(prediction["depth"] - target["depth"]), valid_mask),
    )


def save_point_cloud_ply(
    path: str | Path,
    depth: np.ndarray,
    color: np.ndarray,
    valid_mask: np.ndarray,
    fov_params: np.ndarray,
) -> None:
    """Write a binary PLY without requiring a visualization dependency."""
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    points = depth_to_points(
        np.asarray(depth, dtype=np.float32),
        valid_mask=valid_mask,
        fov_params=np.asarray(fov_params, dtype=np.float32),
    )
    colors = np.asarray(color, dtype=np.uint8)[valid_mask]
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)
