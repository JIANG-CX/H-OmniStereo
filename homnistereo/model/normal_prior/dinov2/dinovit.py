# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 reproduced in
# licenses/DINOV2_LICENSE.

"""Fixed DINOv2 ViT-L/14 encoder used by the heading-normal prior."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .attention import MemEffAttention
from .block import TransformerBlock
from .patch_embed import PatchEmbed


class DinoV2Encoder(nn.Module):
    """Inference-only ViT-L/14 returning four intermediate feature maps."""

    patch_size = 14
    feature_indices = frozenset((5, 11, 17, 23))

    def __init__(self) -> None:
        super().__init__()
        embedding_dim = 1024
        self.patch_embed = PatchEmbed(
            img_size=518,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=embedding_dim,
        )
        patch_count = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count + 1, embedding_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embedding_dim,
                    num_heads=16,
                    mlp_ratio=4,
                    qkv_bias=True,
                    projection_bias=True,
                    feed_forward_bias=True,
                    attention_class=MemEffAttention,
                )
                for _ in range(24)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def _interpolate_position_encoding(
        self,
        tokens: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        original_dtype = tokens.dtype
        patch_position = self.pos_embed[:, 1:].float()
        class_position = self.pos_embed[:, :1].float()
        source_side = int(math.sqrt(patch_position.shape[1]))
        target_height = image_height // self.patch_size
        target_width = image_width // self.patch_size
        patch_position = nn.functional.interpolate(
            patch_position.reshape(1, source_side, source_side, -1).permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode="bicubic",
            antialias=False,
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(
            1,
            target_height * target_width,
            -1,
        )
        return torch.cat([class_position, patch_position], dim=1).to(original_dtype)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        batch_size, _, image_height, image_width = images.shape
        tokens = self.patch_embed(images)
        tokens = torch.cat([self.cls_token.expand(batch_size, -1, -1), tokens], dim=1)
        tokens = tokens + self._interpolate_position_encoding(
            tokens,
            image_height,
            image_width,
        )

        outputs = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self.feature_indices:
                feature = self.norm(tokens)[:, 1:]
                outputs.append(
                    feature.reshape(
                        batch_size,
                        image_height // self.patch_size,
                        image_width // self.patch_size,
                        -1,
                    )
                )
        return outputs
