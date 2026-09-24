import json
from pathlib import Path

import cv2
import pytest

from scripts.infer import load_camera_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets"
EXAMPLES = {
    "360sdnet/stairs": (".png", 0.2, (0.0, 360.0, 0.0, 180.0)),
    "360sdnet/hall": (".png", 0.2, (0.0, 360.0, 0.0, 180.0)),
    "360sdnet/room": (".png", 0.2, (0.0, 360.0, 0.0, 180.0)),
    "threeD60/matterport3d_12": (
        ".png",
        0.2834,
        (0.0, 360.0, 0.0, 180.0),
    ),
    "threeD60/stanford2d3d_area5a": (
        ".png",
        0.2834,
        (0.0, 360.0, 0.0, 180.0),
    ),
    "ours/val21_000094": (
        ".jpg",
        0.2268830066941056,
        (0.0, 360.0, 0.0, 180.0),
    ),
    "ours/val73_000057": (
        ".jpg",
        0.41977808801565253,
        (0.0, 360.0, 0.0, 180.0),
    ),
    "mvsgi/cam1_2_pose_hard_001_000009": (
        ".jpg",
        0.238,
        (90.0, 270.0, 0.0, 180.0),
    ),
    "mvsgi/cam1_2_pose_easy_000_000006": (
        ".jpg",
        0.238,
        (90.0, 270.0, 0.0, 180.0),
    ),
    "mvsgi/cam2_0_pose_easy_003_000019": (
        ".jpg",
        0.30950154135318936,
        (90.0, 270.0, 0.0, 180.0),
    ),
}


@pytest.mark.parametrize(("relative_path", "expected"), EXAMPLES.items())
def test_inference_example_is_complete(relative_path, expected):
    extension, expected_baseline, expected_fov = expected
    example_dir = ASSET_ROOT / relative_path
    top_path = example_dir / f"top{extension}"
    bottom_path = example_dir / f"bottom{extension}"
    camera_path = example_dir / "camera.json"

    top = cv2.imread(str(top_path), cv2.IMREAD_COLOR)
    bottom = cv2.imread(str(bottom_path), cv2.IMREAD_COLOR)
    assert top is not None
    assert bottom is not None
    assert top.shape == bottom.shape

    baseline, fov_params = load_camera_parameters(camera_path)
    assert baseline == expected_baseline
    assert fov_params == expected_fov


def test_camera_files_have_only_the_public_schema():
    for camera_path in ASSET_ROOT.glob("*/*/camera.json"):
        payload = json.loads(camera_path.read_text(encoding="utf-8"))
        assert set(payload) == {"baseline", "fov_params"}

    assert not (ASSET_ROOT / "stairs").exists()


def test_usage_guide_lists_every_inference_example():
    usage_guide = (PROJECT_ROOT / "readme_reader.md").read_text(encoding="utf-8")

    for relative_path in EXAMPLES:
        assert f"`assets/{relative_path}`" in usage_guide
