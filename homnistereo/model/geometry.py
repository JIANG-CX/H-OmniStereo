"""Correlation-volume sampling used by recurrent refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from homnistereo.geometry import amp_safe_normalize
from homnistereo.model.model_utils import bilinear_sampler


class CombinedGeometryEncodingVolume:
    def __init__(self, reference, source, geometry_volume, levels, offsets):
        self.levels = levels
        self.offsets = offsets
        self.geometry_pyramid = []
        self.correlation_pyramid = []

        correlation = self._all_pairs_correlation(reference, source)
        batch, height, width, _, source_width = correlation.shape
        _, channels, _, _, _ = geometry_volume.shape
        geometry_volume = (
            geometry_volume.permute(0, 3, 4, 1, 2)
            .reshape(batch * height * width, channels, 1, -1)
            .contiguous()
        )
        correlation = correlation.reshape(batch * height * width, 1, 1, source_width)
        self.geometry_pyramid.append(geometry_volume)
        self.correlation_pyramid.append(correlation)
        for _ in range(levels - 1):
            geometry_volume = F.avg_pool2d(geometry_volume, [1, 2], stride=[1, 2])
            correlation = F.avg_pool2d(correlation, [1, 2], stride=[1, 2])
            self.geometry_pyramid.append(geometry_volume)
            self.correlation_pyramid.append(correlation)

    def __call__(self, disparity, coordinates, low_memory=False):
        batch, _, height, width = disparity.shape
        offsets = self.offsets.to(disparity.device)
        sampled = []
        for level in range(self.levels):
            local_x = (
                offsets + disparity.reshape(batch * height * width, 1, 1, 1) / 2**level
            )
            local_y = torch.zeros_like(local_x)
            local_coordinates = torch.cat([local_x, local_y], dim=-1)
            geometry = bilinear_sampler(
                self.geometry_pyramid[level], local_coordinates, low_memory=low_memory
            ).reshape(batch, height, width, -1)

            source_x = (
                coordinates.reshape(batch * height * width, 1, 1, 1) / 2**level
                - disparity.reshape(batch * height * width, 1, 1, 1) / 2**level
                + offsets
            )
            source_coordinates = torch.cat([source_x, local_y], dim=-1)
            correlation = bilinear_sampler(
                self.correlation_pyramid[level],
                source_coordinates,
                low_memory=low_memory,
            ).reshape(batch, height, width, -1)
            sampled.extend([geometry, correlation])
        return torch.cat(sampled, dim=-1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _all_pairs_correlation(reference, source):
        batch, channels, height, reference_width = reference.shape
        source_width = source.shape[-1]
        with torch.amp.autocast("cuda", enabled=False):
            correlation = torch.einsum(
                "bchw,bchv->bhwv",
                amp_safe_normalize(reference.float(), dim=1),
                amp_safe_normalize(source.float(), dim=1),
            )
        return correlation.reshape(batch, height, reference_width, 1, source_width)
