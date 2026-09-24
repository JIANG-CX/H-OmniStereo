from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass
class BatchMetrics:
    disparity_mae: float
    disparity_rmse: float
    depth_mae: float
    depth_rmse: float
    d1: float
    bad_1px: float
    bad_2px: float
    bad_3px: float


class BatchWiseMetricAccumulator:
    """Average per-image metrics so every image has equal weight."""

    def __init__(self) -> None:
        self._totals = torch.zeros(8, dtype=torch.float64)
        self._image_count = 0

    def update(
        self,
        predicted_disparity: torch.Tensor,
        target_disparity: torch.Tensor,
        predicted_depth: torch.Tensor,
        target_depth: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        for pred_disp, gt_disp, pred_depth, gt_depth, mask in zip(
            predicted_disparity,
            target_disparity,
            predicted_depth,
            target_depth,
            valid_mask,
        ):
            if not torch.any(mask):
                continue
            disp_error = torch.abs(pred_disp - gt_disp)[mask].float()
            depth_error = torch.abs(pred_depth - gt_depth)[mask].float()
            gt_valid = torch.abs(gt_disp[mask]).float()
            d1 = (
                (disp_error > 3.0) & (disp_error / gt_valid > 0.05)
            ).float().mean() * 100.0
            values = torch.stack(
                [
                    disp_error.mean(),
                    torch.sqrt(torch.mean(disp_error.square())),
                    depth_error.mean(),
                    torch.sqrt(torch.mean(depth_error.square())),
                    d1,
                    (disp_error > 1.0).float().mean() * 100.0,
                    (disp_error > 2.0).float().mean() * 100.0,
                    (disp_error > 3.0).float().mean() * 100.0,
                ]
            )
            self._totals += values.detach().cpu().to(torch.float64)
            self._image_count += 1

    def compute(self) -> dict[str, float]:
        if self._image_count == 0:
            raise RuntimeError(
                "No images with valid ground-truth pixels were evaluated."
            )
        values = (self._totals / self._image_count).tolist()
        return asdict(BatchMetrics(*values))

