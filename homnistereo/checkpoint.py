from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

CHECKPOINT_FORMAT_VERSION = 1
ARCHITECTURE_NAME = "H-OmniStereo"


def _validate_checkpoint_path(checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    with checkpoint_path.open("rb") as stream:
        header = stream.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{checkpoint_path} is a Git LFS pointer, not the checkpoint payload. "
            "Run 'git lfs pull' in the repository."
        )


def _state_dict_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a 'state_dict' mapping.")
    return state_dict


def load_inference_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load the release checkpoint and require an exact architecture match."""
    checkpoint_path = Path(checkpoint_path)
    _validate_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}.")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint.get('format_version')!r}. "
            f"Expected {CHECKPOINT_FORMAT_VERSION}."
        )
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError(
            f"Checkpoint architecture is {checkpoint.get('architecture')!r}, "
            f"expected {ARCHITECTURE_NAME!r}."
        )

    model.load_state_dict(_state_dict_from_checkpoint(checkpoint), strict=True)
    return {
        "format_version": checkpoint["format_version"],
        "architecture": checkpoint["architecture"],
        "max_disparity": int(checkpoint["max_disparity"]),
    }
