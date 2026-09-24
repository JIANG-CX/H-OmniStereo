import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import homnistereo
from homnistereo.predictor import HOmniStereoPredictor
from homnistereo.visualization import save_prediction_visuals
from scripts.infer import build_argument_parser, load_camera_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_predictor_accepts_only_erp_fov_parameters():
    parameters = inspect.signature(HOmniStereoPredictor.predict).parameters

    assert "fov_params" in parameters
    assert "camera" not in parameters
    assert "camera_params" not in parameters


def test_infer_reads_camera_geometry_only_from_json(tmp_path):
    option_strings = {
        option
        for action in build_argument_parser()._actions
        for option in action.option_strings
    }
    assert "--camera-params" in option_strings
    assert "--baseline" not in option_strings
    assert "--fov-params" not in option_strings

    path = tmp_path / "camera.json"
    path.write_text(
        json.dumps({"baseline": 0.2834, "fov_params": [0, 360, 0, 180]}),
        encoding="utf-8",
    )
    baseline, fov_params = load_camera_parameters(path)
    assert baseline == 0.2834
    assert fov_params == (0.0, 360.0, 0.0, 180.0)


def test_every_public_python_module_imports():
    module_names = {
        module.name
        for module in pkgutil.walk_packages(
            homnistereo.__path__, prefix=f"{homnistereo.__name__}."
        )
    }
    module_names.update(
        (
            "scripts.evaluate",
            "scripts.infer",
            "scripts.preprocess_3d60_warp",
            "scripts.preprocess_mvsgi",
        )
    )

    for module_name in sorted(module_names):
        importlib.import_module(module_name)


def test_normal_prior_is_not_saved_or_documented_as_an_output():
    assert "normal" not in inspect.signature(save_prediction_visuals).parameters

    reader = (PROJECT_ROOT / "readme_reader.md").read_text(encoding="utf-8")
    assert "cd /path/to/H-OmniStereo" in reader
    assert "runs without external image paths" not in reader
    assert "heading-aligned normal prior" not in reader


def test_removed_public_surfaces_are_absent():
    removed_paths = (
        "checkpoints/structured3d_heading_normal.pth",
        "homnistereo/data/loaders/fsd.py",
        "homnistereo/data/loaders/helvipad.py",
        "homnistereo/data/loaders/mvsgi_pinhole.py",
        "homnistereo/data/loaders/spatialgen.py",
        "homnistereo/data/loaders/structured3d.py",
        "homnistereo/data/loaders/tartanair_v2.py",
    )

    assert not any((PROJECT_ROOT / path).exists() for path in removed_paths)
    assert (PROJECT_ROOT / "readme_reader.md").is_file()
