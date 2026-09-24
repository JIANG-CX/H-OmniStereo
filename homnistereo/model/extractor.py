"""Image and heading-aligned normal-prior feature encoders."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from homnistereo.model.normal_prior import HeadingAlignedNormalPrior
from homnistereo.model.submodule import BasicConv, Conv2x_IN, LayerNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        normalizers = {
            "group": lambda: nn.GroupNorm(num_groups=num_groups, num_channels=planes),
            "batch": lambda: nn.BatchNorm2d(planes),
            "instance": lambda: nn.InstanceNorm2d(planes),
            "layer": lambda: LayerNorm2d(planes),
            "none": nn.Sequential,
        }
        make_norm = normalizers[norm_fn]
        self.norm1 = make_norm()
        self.norm2 = make_norm()
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.norm3 = make_norm()
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, inputs):
        residual = inputs
        output = self.relu(self.norm1(self.conv1(inputs)))
        output = self.relu(self.norm2(self.conv2(output)))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(residual + output)


class ContextNet(nn.Module):
    """Three-level context encoder with fused normal-prior features."""

    def __init__(self, args, output_dim, norm_fn="batch", downsample=2):
        super().__init__()
        self.out_dims = output_dim
        self.norm_fn = norm_fn
        if norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        elif norm_fn == "layer":
            self.norm1 = LayerNorm2d(64)
        else:
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        self.conv2 = BasicConv(
            128 + args.prior_feature_dim, 128, kernel_size=3, padding=1
        )
        self.outputs04 = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(128, 128, norm_fn, stride=1),
                    nn.Conv2d(128, dimensions[2], 3, padding=1),
                )
                for dimensions in output_dim
            ]
        )
        self.outputs08 = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(128, 128, norm_fn, stride=1),
                    nn.Conv2d(128, dimensions[1], 3, padding=1),
                )
                for dimensions in output_dim
            ]
        )
        self.outputs16 = nn.ModuleList(
            [nn.Conv2d(128, dimensions[0], 3, padding=1) for dimensions in output_dim]
        )

    def _make_layer(self, dimensions, stride=1):
        layers = (
            ResidualBlock(self.in_planes, dimensions, self.norm_fn, stride=stride),
            ResidualBlock(dimensions, dimensions, self.norm_fn, stride=1),
        )
        self.in_planes = dimensions
        return nn.Sequential(*layers)

    def forward(self, image, prior_features, num_layers=3):
        output = self.relu1(self.norm1(self.conv1(image)))
        output = self.layer3(self.layer2(self.layer1(output)))
        output = self.conv2(torch.cat([output, prior_features], dim=1))
        outputs04 = [head(output) for head in self.outputs04]
        output08 = self.layer4(output)
        outputs08 = [head(output08) for head in self.outputs08]
        output16 = self.layer5(output08)
        outputs16 = [head(output16) for head in self.outputs16]
        return outputs04, outputs08, outputs16


class HybridFeatureEncoder(nn.Module):
    """Fuse EdgeNeXt image features with heading-aligned normal-prior features."""

    def __init__(self, args):
        super().__init__()
        image_backbone = timm.create_model(
            "edgenext_small",
            pretrained=False,
            features_only=False,
            in_chans=3,
        )
        self.stem = image_backbone.stem
        self.stages = image_backbone.stages
        self.heading_normal_prior = HeadingAlignedNormalPrior()
        for parameter in self.heading_normal_prior.parameters():
            parameter.requires_grad_(False)

        channels = [48, 96, 160, 304]
        self.deconv32_16 = Conv2x_IN(channels[3], channels[2], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(
            channels[2] * 2, channels[1], deconv=True, concat=True
        )
        self.deconv8_4 = Conv2x_IN(
            channels[1] * 2, channels[0], deconv=True, concat=True
        )
        fused_channels = channels[0] * 2 + args.prior_feature_dim
        self.conv4 = nn.Sequential(
            BasicConv(
                fused_channels,
                fused_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm="instance",
            ),
            ResidualBlock(fused_channels, fused_channels, norm_fn="instance"),
            ResidualBlock(fused_channels, fused_channels, norm_fn="instance"),
        )
        self.d_out = [fused_channels, channels[1] * 2, channels[2] * 2, channels[3]]

    def forward(self, images, fov_params):
        _, _, height, width = images.shape
        prior_images = torch.rot90(images, k=1, dims=(-2, -1))
        prior_batch = {
            "image": prior_images,
            "fov_params": fov_params,
        }
        with torch.inference_mode():
            prior_output = self.heading_normal_prior(prior_batch)
        prior_features = torch.rot90(prior_output["features"], k=-1, dims=(-2, -1))
        heading_normal = torch.rot90(prior_output["normal"], k=-1, dims=(-2, -1))
        prior_features = F.interpolate(
            prior_features,
            size=(height // 4, width // 4),
            mode="bilinear",
            align_corners=True,
        )

        output04 = self.stages[0](self.stem(images))
        output08 = self.stages[1](output04)
        output16 = self.stages[2](output08)
        output32 = self.stages[3](output16)
        output16 = self.deconv32_16(output32, output16)
        output08 = self.deconv16_8(output16, output08)
        output04 = self.deconv8_4(output08, output04)
        output04 = self.conv4(torch.cat([output04, prior_features], dim=1))
        return [output04, output08, output16, output32], prior_features, heading_normal
