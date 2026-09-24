# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 reproduced in
# licenses/DINOV2_LICENSE.

"""Inference-only DINOv2 transformer block."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .attention import Attention
from .layer_scale import LayerScale
from .mlp import Mlp


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        projection_bias: bool,
        feed_forward_bias: bool,
        attention_class: Callable[..., nn.Module] = Attention,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=projection_bias,
        )
        self.ls1 = LayerScale(dim, init_values=1.0)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            bias=feed_forward_bias,
        )
        self.ls2 = LayerScale(dim, init_values=1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.ls1(self.attn(self.norm1(inputs)))
        return inputs + self.ls2(self.mlp(self.norm2(inputs)))
