# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from homnistereo.geometry import amp_safe_normalize


def _is_contiguous(tensor: torch.Tensor) -> bool:
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    r"""https://huggingface.co/spaces/Roll20/pet_score/blob/b258ef28152ab0d5b377d9142a23346f863c1526/lib/timm/models/convnext.py#L85
    LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """
        @x: (B,C,H,W)
        """
        if _is_contiguous(x):
            return (
                F.layer_norm(
                    x.permute(0, 2, 3, 1),
                    self.normalized_shape,
                    self.weight,
                    self.bias,
                    self.eps,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        bn=True,
        relu=True,
        norm="batch",
        **kwargs,
    ):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        self.bn = nn.Identity()
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                if norm == "batch":
                    self.bn = nn.BatchNorm3d(out_channels)
                elif norm == "instance":
                    self.bn = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                if norm == "batch":
                    self.bn = nn.BatchNorm2d(out_channels)
                elif norm == "instance":
                    self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class Conv3dNormActReduced(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        hidden=None,
        kernel_size=3,
        kernel_disp=None,
        stride=1,
        norm=nn.BatchNorm3d,
    ):
        super().__init__()
        if kernel_disp is None:
            kernel_disp = kernel_size
        if hidden is None:
            hidden = C_out
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                C_in,
                hidden,
                kernel_size=(1, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2),
                stride=(1, stride, stride),
            ),
            norm(hidden),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                hidden,
                C_out,
                kernel_size=(kernel_disp, 1, 1),
                padding=(kernel_disp // 2, 0, 0),
                stride=(stride, 1, 1),
            ),
            norm(C_out),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        @x: (B,C,D,H,W)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResnetBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        padding=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super().__init__()
        self.norm_layer = norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=padding,
        )
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=padding,
        )
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResnetBasicBlock3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        padding=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm3d,
        bias=False,
    ):
        super().__init__()
        self.norm_layer = norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=padding,
        )
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=padding,
        )
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        """Apply scaled dot-product attention to ``[batch, tokens, channels]``."""
        batch_size, token_count, _ = query.shape
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Convert [B, L, H, D] to the layout expected by PyTorch SDPA.
        query = query.view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attn_output = (
            F.scaled_dot_product_attention(query, key, value)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        attn_output = attn_output.reshape(batch_size, token_count, -1)
        output = self.out_proj(attn_output)

        return output


class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout=0.1,
        act=nn.GELU,
        norm=nn.LayerNorm,
    ):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class Conv2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        concat=True,
        keep_concat=True,
        bn=True,
        relu=True,
        keep_dispc=False,
    ):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                bn=bn,
                relu=True,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = BasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                bn=bn,
                relu=True,
                kernel_size=kernel,
                stride=2,
                padding=1,
            )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(
                out_channels * 2,
                out_channels * mul,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = BasicConv(
                out_channels,
                out_channels,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="bilinear")
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        IN=True,
        relu=True,
        **kwargs,
    ):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class Conv2x_IN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        concat=True,
        keep_concat=True,
        IN=True,
        relu=True,
        keep_dispc=False,
    ):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                IN=True,
                relu=True,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = BasicConv_IN(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                IN=True,
                relu=True,
                kernel_size=kernel,
                stride=2,
                padding=1,
            )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = ResnetBasicBlock(
                out_channels * 2,
                out_channels * mul,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.InstanceNorm2d,
            )
        else:
            self.conv2 = BasicConv_IN(
                out_channels,
                out_channels,
                False,
                is_3d,
                IN,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="bilinear")
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(reference, source, groups):
    batch_size, channels, height, width = reference.shape
    if channels % groups:
        raise ValueError(f"{channels} channels cannot be divided into {groups} groups.")
    channels_per_group = channels // groups
    reference = reference.reshape(batch_size, groups, channels_per_group, height, width)
    source = source.reshape(batch_size, groups, channels_per_group, height, width)
    with torch.amp.autocast("cuda", enabled=False):
        # Normalize each group before summing to keep half-precision inputs stable.
        correlation = (
            amp_safe_normalize(reference.float(), dim=2, eps=1e-4)
            * amp_safe_normalize(source.float(), dim=2, eps=1e-4)
        ).sum(dim=2)
    return correlation


def build_groupwise_correlation_volume(reference, source, max_disparity, groups):
    """Build the fixed left-reference horizontal correlation volume."""
    batch_size, _, height, width = reference.shape
    volume = reference.new_zeros([batch_size, groups, max_disparity, height, width])
    for disparity in range(max_disparity):
        if disparity == 0:
            volume[:, :, disparity] = groupwise_correlation(reference, source, groups)
            continue
        volume[:, :, disparity, :, disparity:] = groupwise_correlation(
            reference[:, :, :, disparity:], source[:, :, :, :-disparity], groups
        )
    return volume.contiguous()


def build_concatenation_volume(reference, source, max_disparity):
    """Build the fixed left-reference horizontal concatenation volume."""
    batch_size, channels, height, width = reference.shape
    volume = reference.new_zeros(
        [batch_size, 2 * channels, max_disparity, height, width]
    )
    for disparity in range(max_disparity):
        volume[:, :channels, disparity] = reference
        if disparity == 0:
            volume[:, channels:, disparity] = source
            continue
        volume[:, channels:, disparity, :, disparity:] = source[:, :, :, :-disparity]
    return volume.contiguous()


def regress_disparity(probability, max_disparity):
    """Compute the expectation of a disparity probability volume."""
    if probability.ndim != 4:
        raise ValueError("Disparity probability must have shape [B, D, H, W].")
    disparities = torch.arange(
        max_disparity, dtype=probability.dtype, device=probability.device
    ).reshape(1, max_disparity, 1, 1)
    return torch.sum(probability * disparities, dim=1, keepdim=True)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
        )

    def forward(self, cv, feat):
        """
        @cv: cost volume (B,C,D,H,W)
        @feat: (B,C,H,W)
        """
        feat_att = self.feat_att(feat).unsqueeze(2)  # (B,C,1,H,W)
        cv = torch.sigmoid(feat_att) * cv
        return cv


def context_upsample(feat_low, up_weights):
    """
    @feat_low: (b, c, h, w)  1/4 resolution
    @up_weights: (b, 9, 4*h, 4*w)  Image resolution
    """
    b, c, h, w = feat_low.shape

    # Unfold each channel: (b, c, 9, h, w)
    feat_unfold = F.unfold(feat_low, 3, 1, 1).reshape(b, c, 9, h, w)
    # Upsample to target resolution: (b, c, 9, 4h, 4w)
    feat_unfold = F.interpolate(
        feat_unfold.reshape(b, c * 9, h, w), (h * 4, w * 4), mode="nearest"
    ).reshape(b, c, 9, h * 4, w * 4)

    # Weighted sum over 9 neighbors: (b, c, 4h, 4w)
    # up_weights is (b, 9, 4h, 4w), we expand to (b, 1, 9, 4h, 4w)
    feat = (feat_unfold * up_weights.unsqueeze(1)).sum(2)

    return feat


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (N,1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        ).exp()[None]

        pe[:, 0::2] = torch.sin(position * div_term)  # (N, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x, resize_embed=False):
        """
        @x: (B,N,D)
        """
        self.pe = self.pe.to(x.device).to(x.dtype)
        pe = self.pe
        if pe.shape[1] < x.shape[1]:
            if resize_embed:
                pe = F.interpolate(
                    pe.permute(0, 2, 1),
                    size=x.shape[1],
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            else:
                raise RuntimeError(f"x:{x.shape}, pe:{pe.shape}")
        return x + pe[:, : x.size(1)]


class CostVolumeDisparityAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        act=nn.GELU,
        norm_first=False,
        num_transformer=6,
        max_len=512,
        resize_embed=False,
    ):
        super().__init__()
        self.resize_embed = resize_embed
        self.sa = nn.ModuleList([])
        for _ in range(num_transformer):
            self.sa.append(
                FlashAttentionTransformerEncoderLayer(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dim_feedforward=dim_feedforward,
                    act=act,
                    dropout=dropout,
                )
            )
        self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)

    def forward(self, cost_volume):
        """Attend along disparity for a ``[B, C, D, H, W]`` cost volume."""
        batch_size, channels, disparities, height, width = cost_volume.shape
        x = cost_volume.permute(0, 3, 4, 2, 1).reshape(
            batch_size * height * width, disparities, channels
        )
        x = self.pos_embed0(x, resize_embed=self.resize_embed)
        for attention_layer in self.sa:
            x = attention_layer(x)
        x = x.reshape(batch_size, height, width, disparities, channels).permute(
            0, 4, 3, 1, 2
        )

        return x


class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7, in_channels=2):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(
            in_channels, 1, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guidance=None):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        if guidance is not None:
            x = torch.cat([avg_out, max_out, guidance], dim=1)
        else:
            x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)


class EdgeNextConvEncoder(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale_init_value=1e-6,
        expan_ratio=4,
        kernel_size=7,
        norm="layer",
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        if norm == "layer":
            self.norm = LayerNorm2d(dim, eps=1e-6)
        else:
            self.norm = nn.Identity()
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
