"""Spherical heading-aligned normal prior used by H-OmniStereo."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from homnistereo.geometry import get_rays, get_resize_keep_aspect_ratio, normalize_image

from .dinov2 import DinoV2Encoder
from .spherical_decoder import SphericalNormalDecoder


class HeadingAlignedNormalPrior(nn.Module):
    """Extract spherical normal-prior features from one ERP image."""

    def __init__(self):
        super().__init__()
        self.backbone = DinoV2Encoder()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.spherical_decoder = SphericalNormalDecoder()

    def forward(self, batch):
        images = batch["image"]
        batch_size, _, raw_height, raw_width = images.shape
        fov_params = batch["fov_params"].clone()
        longitude_span = fov_params[:, 1] - fov_params[:, 0]
        fov_params[:, 0] = 180.0 - longitude_span / 2.0
        fov_params[:, 1] = 180.0 + longitude_span / 2.0

        height, width = get_resize_keep_aspect_ratio(
            raw_height,
            raw_width,
            divider=self.backbone.patch_size,
            max_H=1344,
            max_W=1344,
        )
        resized_images = F.interpolate(
            normalize_image(images),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        spherical_rays = get_rays(
            batch_size,
            height,
            width,
            fov_params,
            only_lat=False,
        ).to(images.device)

        with torch.inference_mode():
            backbone_features = [
                feature.contiguous() for feature in self.backbone(resized_images)
            ]
        normal, prior_features = self.spherical_decoder(
            resized_images,
            backbone_features,
            spherical_rays,
        )
        normal = F.interpolate(
            normal, size=(raw_height, raw_width), mode="bilinear", align_corners=False
        )
        return {"normal": normal, "features": prior_features}
