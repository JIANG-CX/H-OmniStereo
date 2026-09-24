from pathlib import Path

import torch
import pytest

from homnistereo.checkpoint import load_inference_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_checkpoint_has_inference_only_schema_and_names():
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints/h_omnistereo.pth",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    assert set(checkpoint) == {
        "format_version",
        "architecture",
        "max_disparity",
        "state_dict",
    }
    assert checkpoint["architecture"] == "H-OmniStereo"
    assert checkpoint["max_disparity"] == 256
    keys = tuple(checkpoint["state_dict"])
    assert len(keys) == 1451
    assert any(key.startswith("feature_encoder.heading_normal_prior.") for key in keys)
    assert not any(
        legacy_name in key
        for key in keys
        for legacy_name in ("unik3d_model", "vit_w_esphere", "_vit_w_esphere")
    )
    assert not any(
        unused_parameter in key
        for key in keys
        for unused_parameter in ("register_tokens", "mask_token")
    )
    assert not any(key.startswith(("optimizer", "lr_scheduler")) for key in keys)


def test_git_lfs_pointer_has_actionable_error(tmp_path):
    pointer = tmp_path / "model.pth"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 123\n",
        encoding="ascii",
    )

    with pytest.raises(RuntimeError, match="git lfs pull"):
        load_inference_checkpoint(torch.nn.Identity(), pointer)
