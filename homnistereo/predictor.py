"""High-level ERP stereo inference interface shared by the command-line tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from homnistereo.checkpoint import load_inference_checkpoint
from homnistereo.config import ModelConfig
from homnistereo.geometry import disp_to_depth
from homnistereo.model import HOmniStereo


@dataclass(frozen=True)
class Prediction:
    disparity: torch.Tensor
    depth: torch.Tensor
    uncertainty: torch.Tensor
    heading_normal: torch.Tensor
    valid_mask: torch.Tensor


class HOmniStereoPredictor:
    """Load the release checkpoint and predict one or more ERP stereo pairs."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
        iterations: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for H-OmniStereo inference, but no CUDA device is available."
            )
        self.config = ModelConfig()
        self.iterations = (
            self.config.inference_iterations if iterations is None else iterations
        )
        if self.iterations <= 0:
            raise ValueError("Inference iterations must be positive.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = HOmniStereo(self.config)
        metadata = load_inference_checkpoint(self.model, checkpoint)
        if metadata["max_disparity"] != self.config.max_disparity:
            raise ValueError(
                f"Checkpoint max disparity is {metadata['max_disparity']}, "
                f"but the release architecture requires {self.config.max_disparity}."
            )
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _validate_inputs(
        top_image: torch.Tensor,
        bottom_image: torch.Tensor,
        baseline: torch.Tensor,
        fov_params: torch.Tensor,
    ) -> None:
        if top_image.ndim != 4 or top_image.shape[1] != 3:
            raise ValueError("top_image must have shape [B, 3, H, W].")
        if top_image.shape != bottom_image.shape:
            raise ValueError("Top and bottom image tensors must have identical shapes.")
        if top_image.shape[-2] % 32 or top_image.shape[-1] % 32:
            raise ValueError("Image height and width must both be divisible by 32.")
        batch_size = top_image.shape[0]
        if baseline.numel() != batch_size:
            raise ValueError("baseline must contain one value per stereo pair.")
        if fov_params.shape != (batch_size, 4):
            raise ValueError("fov_params must have shape [B, 4].")
        if torch.any(baseline <= 0):
            raise ValueError("Every baseline must be positive.")
        if torch.any(fov_params[:, 1] <= fov_params[:, 0]) or torch.any(
            fov_params[:, 3] <= fov_params[:, 2]
        ):
            raise ValueError("ERP longitude and latitude ranges must be increasing.")

    @torch.inference_mode()
    def predict(
        self,
        top_image: torch.Tensor,
        bottom_image: torch.Tensor,
        baseline: torch.Tensor,
        fov_params: torch.Tensor,
        depth_mask: torch.Tensor | None = None,
    ) -> Prediction:
        """Predict bottom-view disparity and radial depth for ERP stereo pairs."""
        self._validate_inputs(top_image, bottom_image, baseline, fov_params)
        batch_size = top_image.shape[0]
        top_image = top_image.to(self.device, non_blocking=True).float()
        bottom_image = bottom_image.to(self.device, non_blocking=True).float()
        baseline = baseline.to(self.device, non_blocking=True).float().reshape(batch_size)
        fov_params = fov_params.to(self.device, non_blocking=True).float()
        if depth_mask is not None:
            depth_mask = depth_mask.to(self.device, non_blocking=True).bool()

        # The horizontal backbone receives a clockwise-rotated, pair-swapped view.
        # Rotating the output back preserves the original bottom-view reference.
        internal_batch = {
            "reference_image": torch.rot90(bottom_image, k=-1, dims=(-2, -1)),
            "source_image": torch.rot90(top_image, k=-1, dims=(-2, -1)),
            "fov_params": fov_params,
        }
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda" and self.config.mixed_precision,
        ):
            output = self.model(internal_batch, iterations=self.iterations)

        internal_disparity = output["disparity"][:, 0]
        internal_x = (
            torch.arange(
                internal_disparity.shape[-1],
                device=self.device,
                dtype=internal_disparity.dtype,
            ).view(1, 1, -1)
            + 0.5
        )
        internal_invalid = internal_x < internal_disparity

        disparity = torch.rot90(internal_disparity, k=1, dims=(-2, -1))
        invalid = torch.rot90(internal_invalid, k=1, dims=(-2, -1))
        log_uncertainty = torch.rot90(
            output["log_uncertainty"][:, 0], k=1, dims=(-2, -1)
        )
        heading_normal = torch.rot90(output["heading_normal"], k=1, dims=(-2, -1))
        valid_mask = (~invalid) & torch.isfinite(disparity) & (disparity > 1e-4)
        depth_valid_mask = (
            depth_mask
            if depth_mask is not None
            else torch.isfinite(disparity) & (disparity > 1e-4)
        )
        depth = disp_to_depth(
            disparity,
            baseline,
            depth_valid_mask,
            fov_params=fov_params,
        )
        return Prediction(
            disparity=disparity,
            depth=depth,
            uncertainty=torch.exp(log_uncertainty.clamp(-20.0, 20.0)),
            heading_normal=heading_normal,
            valid_mask=valid_mask,
        )
