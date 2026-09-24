from pathlib import Path

from homnistereo.data.dataset import load_dataset_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_registry_contains_only_erp_stereo_datasets():
    registry = load_dataset_registry(PROJECT_ROOT / "configs/datasets.json")

    assert registry["default_dataset"] == "threeD60"
    assert set(registry["datasets"]) == {
        "threeD60",
        "threeD60warp",
        "mvsgi",
        "ours",
        "deep360",
    }
    for specification in registry["datasets"].values():
        assert "aliases" not in specification
        assert "camera" not in specification
        assert "normal_ground_truth" not in specification
        assert "normal_only" not in specification
        assert "normal_checkpoint" not in specification


def test_paper_disparity_splits_are_fixed():
    registry = load_dataset_registry(PROJECT_ROOT / "configs/datasets.json")

    warp = registry["datasets"]["threeD60warp"]
    assert Path(warp["root"]).name == "3D60_Warp_new0413"
    assert warp["split"] == "test_index.txt"


def test_public_registry_contains_no_machine_specific_paths():
    registry_text = (PROJECT_ROOT / "configs/datasets.json").read_text(encoding="utf-8")

    assert "/home/" not in registry_text
    assert "/media/" not in registry_text
