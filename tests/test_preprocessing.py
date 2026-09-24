import hashlib
from pathlib import Path

import numpy as np
import pytest

from homnistereo.data.loaders.common import decode_float32_rgba
from homnistereo.preprocessing import create_new_output_root, read_split_entries
from scripts.preprocess_3d60_warp import build_argument_parser as build_3d60_parser
from scripts.preprocess_mvsgi import (
    build_argument_parser as build_mvsgi_parser,
    encode_float32_rgba,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "splits/threeD60/test_index.txt": (
        2189,
        "32d2517eb6471c1f06726fbd396e8c18dbd32862d8718bc1e4ad7e90a347405a",
    ),
    "splits/threeD60warp/test_index.txt": (
        2189,
        "32d2517eb6471c1f06726fbd396e8c18dbd32862d8718bc1e4ad7e90a347405a",
    ),
    "splits/mvsgi/test_index.txt": (
        26339,
        "9247c51a3bd0ace15c8a88384c5caacb45640d82015927471fe0110423d19704",
    ),
}


@pytest.mark.parametrize(("relative_path", "expected"), SPLITS.items())
def test_release_split_is_complete(relative_path, expected):
    expected_count, expected_sha256 = expected
    path = PROJECT_ROOT / relative_path
    assert len(read_split_entries(path)) == expected_count
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def test_preprocessing_output_must_be_new_and_outside_input(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()

    with pytest.raises(ValueError, match="outside the input root"):
        create_new_output_root(input_root, input_root / "generated")

    existing_output = tmp_path / "existing"
    existing_output.mkdir()
    with pytest.raises(FileExistsError, match="avoid overwriting"):
        create_new_output_root(input_root, existing_output)

    new_output = tmp_path / "new-output"
    assert create_new_output_root(input_root, new_output) == new_output
    assert new_output.is_dir()


def test_mvsgi_float32_depth_encoding_round_trip():
    depth = np.asarray([[1.25, -1.0], [127.5, 0.0]], dtype=np.float32)

    decoded = decode_float32_rgba(encode_float32_rgba(depth))

    np.testing.assert_array_equal(decoded, depth)


@pytest.mark.parametrize("build_parser", (build_3d60_parser, build_mvsgi_parser))
def test_preprocessing_cli_has_no_overwrite_mode(build_parser):
    parser = build_parser()
    actions = {option for action in parser._actions for option in action.option_strings}

    assert "--input-root" in actions
    assert "--output-root" in actions
    assert "--overwrite" not in actions


def test_usage_guide_documents_preprocessing_and_acknowledgements():
    usage_guide = (PROJECT_ROOT / "readme_reader.md").read_text(encoding="utf-8")

    for path in SPLITS:
        assert f"`{path}`" in usage_guide
    for script in ("preprocess_3d60_warp.py", "preprocess_mvsgi.py"):
        assert f"scripts/{script}" in usage_guide
    for repository in (
        "https://github.com/NVlabs/FoundationStereo/",
        "https://github.com/EnVision-Research/DA-2",
        "https://github.com/VCL3D/3D60",
        "https://github.com/castacks/mvs_gi",
    ):
        assert repository in usage_guide
