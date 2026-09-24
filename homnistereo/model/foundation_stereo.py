"""H-OmniStereo inference network.

This module contains only the architecture used by the released checkpoint.
Training losses, alternative feature encoders, and model-selection branches are
intentionally excluded from the public inference package.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from homnistereo.config import ModelConfig
from homnistereo.geometry import normalize_image
from homnistereo.model.extractor import ContextNet, HybridFeatureEncoder
from homnistereo.model.geometry import CombinedGeometryEncodingVolume
from homnistereo.model.submodule import (
    BasicConv,
    BasicConv_IN,
    ChannelAttentionEnhancement,
    Conv2x,
    Conv3dNormActReduced,
    CostVolumeDisparityAttention,
    FeatureAtt,
    ResnetBasicBlock3D,
    SpatialAttentionExtractor,
    build_concatenation_volume,
    build_groupwise_correlation_volume,
    context_upsample,
    regress_disparity,
)
from homnistereo.model.update import RecurrentUpdateBlock


class CostAggregationHourglass(nn.Module):
    def __init__(
        self, config: ModelConfig, in_channels: int, feature_dims: list[int]
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv(
                in_channels,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(
                in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17
            ),
        )
        self.conv2 = nn.Sequential(
            BasicConv(
                in_channels * 2,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(
                in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17
            ),
        )
        self.conv3 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 6,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(
                in_channels * 6, in_channels * 6, kernel_size=3, kernel_disp=17
            ),
        )
        self.conv3_up = BasicConv(
            in_channels * 6,
            in_channels * 4,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv2_up = BasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv1_up = BasicConv(
            in_channels * 2,
            in_channels,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv_out = nn.Sequential(
            Conv3dNormActReduced(
                in_channels, in_channels, kernel_size=3, kernel_disp=17
            ),
            Conv3dNormActReduced(
                in_channels, in_channels, kernel_size=3, kernel_disp=17
            ),
        )
        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1),
            Conv3dNormActReduced(
                in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17
            ),
            Conv3dNormActReduced(
                in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17
            ),
        )
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1),
            Conv3dNormActReduced(
                in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17
            ),
            Conv3dNormActReduced(
                in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17
            ),
        )
        self.atts = nn.ModuleDict(
            {
                "4": CostVolumeDisparityAttention(
                    d_model=in_channels,
                    nhead=4,
                    dim_feedforward=in_channels,
                    norm_first=False,
                    num_transformer=4,
                    max_len=config.max_disparity // 16,
                )
            }
        )
        self.conv_patch = nn.Sequential(
            nn.Conv3d(
                in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels
            ),
            nn.BatchNorm3d(in_channels),
        )
        self.feature_att_8 = FeatureAtt(in_channels * 2, feature_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels * 4, feature_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels * 6, feature_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, feature_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, feature_dims[1])

    @staticmethod
    def _align(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-3:] == reference.shape[-3:]:
            return inputs
        return F.interpolate(
            inputs, size=reference.shape[-3:], mode="trilinear", align_corners=False
        )

    def forward(
        self, volume: torch.Tensor, features: list[torch.Tensor]
    ) -> torch.Tensor:
        conv1 = self.feature_att_8(self.conv1(volume), features[1])
        conv2 = self.feature_att_16(self.conv2(conv1), features[2])
        conv3 = self.feature_att_32(self.conv3(conv2), features[3])

        conv2 = torch.cat((self._align(self.conv3_up(conv3), conv2), conv2), dim=1)
        conv2 = self.feature_att_up_16(self.agg_0(conv2), features[2])
        conv1 = torch.cat((self._align(self.conv2_up(conv2), conv1), conv1), dim=1)
        conv1 = self.feature_att_up_8(self.agg_1(conv1), features[1])

        output = self.conv1_up(conv1)
        residual = self.atts["4"](self.conv_patch(volume))
        residual = F.interpolate(
            residual, scale_factor=4, mode="trilinear", align_corners=False
        )
        return self.conv_out(output + self._align(residual, output))


class HOmniStereo(nn.Module):
    """Fixed H-OmniStereo model used for public evaluation and inference."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        context_dims = self.config.hidden_dims
        volume_dim = 28

        self.cnet = ContextNet(
            self.config,
            output_dim=[self.config.hidden_dims, context_dims],
            downsample=self.config.n_downsample,
        )
        self.update_block = RecurrentUpdateBlock(
            self.config,
            self.config.hidden_dims[0],
            volume_dim=volume_dim,
        )
        self.sam = SpatialAttentionExtractor(in_channels=2)
        self.cam = ChannelAttentionEnhancement(self.config.hidden_dims[0])
        self.context_zqr_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    context_dims[index],
                    self.config.hidden_dims[index] * 3,
                    kernel_size=3,
                    padding=1,
                )
                for index in range(self.config.n_gru_layers)
            ]
        )
        self.feature_encoder = HybridFeatureEncoder(self.config)
        self.proj_cmb = nn.Conv2d(self.feature_encoder.d_out[0], 12, kernel_size=1)
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(64, 9, kernel_size=4, stride=2, padding=1)
        )
        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(
                volume_dim, volume_dim, kernel_size=3, stride=1, padding=1
            ),
            ResnetBasicBlock3D(
                volume_dim, volume_dim, kernel_size=3, stride=1, padding=1
            ),
        )
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature_encoder.d_out[0])
        self.cost_agg = CostAggregationHourglass(
            self.config,
            in_channels=volume_dim,
            feature_dims=self.feature_encoder.d_out,
        )
        self.classifier = nn.Sequential(
            BasicConv(
                volume_dim, volume_dim // 2, kernel_size=3, padding=1, is_3d=True
            ),
            ResnetBasicBlock3D(
                volume_dim // 2, volume_dim // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv3d(volume_dim // 2, 1, kernel_size=7, padding=3),
        )
        self.cov_spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.cov_spx_gru = nn.Sequential(
            nn.ConvTranspose2d(64, 9, kernel_size=4, stride=2, padding=1)
        )
        radius = self.config.corr_radius
        offsets = torch.linspace(-radius, radius, 2 * radius + 1).reshape(1, 1, -1, 1)
        self.register_buffer("dx", offsets, persistent=False)

    def _upsample_disparity(self, disparity, upsampling_features, stem_features):
        with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
            weights = F.softmax(
                self.spx_gru(self.spx_2_gru(upsampling_features, stem_features)),
                dim=1,
            )
            output = context_upsample(disparity * 4.0, weights)
        return output.float()

    def _upsample_log_uncertainty(
        self, uncertainty, upsampling_features, stem_features
    ):
        with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
            weights = F.softmax(
                self.cov_spx_gru(
                    self.cov_spx_2_gru(upsampling_features, stem_features)
                ),
                dim=1,
            )
            output = context_upsample(uncertainty, weights) + math.log(4.0)
        return output.float()

    def forward(self, batch: dict[str, object], iterations: int | None = None):
        """Predict bottom-view disparity from a preprocessed vertical stereo pair."""
        reference = batch["reference_image"]
        source = batch["source_image"]
        if not isinstance(reference, torch.Tensor) or not isinstance(
            source, torch.Tensor
        ):
            raise TypeError("reference_image and source_image must be tensors")
        batch_size, _, _, _ = reference.shape
        iterations = (
            self.config.inference_iterations if iterations is None else iterations
        )
        if iterations <= 0:
            raise ValueError("Inference iterations must be positive.")
        device = reference.device
        reference = normalize_image(reference)
        source = normalize_image(source)

        fov_params = batch["fov_params"]
        if not isinstance(fov_params, torch.Tensor):
            raise TypeError("fov_params must be a tensor")
        feature_fov_params = torch.cat([fov_params, fov_params], dim=0)

        with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
            all_features, prior_features, all_heading_normals = self.feature_encoder(
                torch.cat([reference, source], dim=0),
                feature_fov_params,
            )
            reference_features = [features[:batch_size] for features in all_features]
            source_features = [features[batch_size:] for features in all_features]
            prior_features = prior_features[:batch_size].clone()
            heading_normal = all_heading_normals[:batch_size]
            stem_features = self.stem_2(reference)

            groupwise_volume = build_groupwise_correlation_volume(
                reference_features[0],
                source_features[0],
                self.config.max_disparity // 4,
                8,
            )
            reference_projection = self.proj_cmb(reference_features[0])
            source_projection = self.proj_cmb(source_features[0])
            concatenation_volume = build_concatenation_volume(
                reference_projection,
                source_projection,
                max_disparity=self.config.max_disparity // 4,
            )
            geometry_volume = self.corr_stem(
                torch.cat([groupwise_volume, concatenation_volume], dim=1)
            )
            geometry_volume = self.corr_feature_att(
                geometry_volume, reference_features[0]
            )
            geometry_volume = self.cost_agg(geometry_volume, reference_features)
            probability = F.softmax(self.classifier(geometry_volume).squeeze(1), dim=1)
            disparity = regress_disparity(
                probability, self.config.max_disparity // 4
            ).float()

            context_pyramid = list(
                self.cnet(
                    reference, prior_features, num_layers=self.config.n_gru_layers
                )
            )
            hidden = [torch.tanh(level[0]) for level in context_pyramid]
            context = [torch.relu(level[1]) for level in context_pyramid]
            context = [self.cam(level) * level for level in context]
            attention = [self.sam(level) for level in context]

        geometry_sampler = CombinedGeometryEncodingVolume(
            reference_features[0].float(),
            source_features[0].float(),
            geometry_volume.float(),
            levels=self.config.corr_levels,
            offsets=self.dx,
        )
        _, _, height, width = reference_features[0].shape
        coordinates = torch.arange(width, dtype=torch.float32, device=device)
        coordinates = coordinates.reshape(1, 1, width, 1).repeat(
            batch_size, height, 1, 1
        )
        uncertainty_hidden = hidden[0].clone()
        log_uncertainty = torch.zeros_like(disparity)

        for _ in range(iterations):
            disparity = disparity.detach()
            correlation = geometry_sampler(
                disparity,
                coordinates,
                low_memory=self.config.low_memory,
            )
            with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
                update = self.update_block(
                    hidden,
                    context,
                    correlation,
                    disparity,
                    attention,
                    uncertainty_hidden,
                )
            hidden = update["hidden"]
            uncertainty_hidden = update["uncertainty_hidden"]
            disparity = disparity + update["disparity_delta"].float()
            log_uncertainty = update["log_uncertainty"]

        disparity = self._upsample_disparity(
            disparity,
            update["upsampling_features"].float(),
            stem_features.float(),
        )
        log_uncertainty = self._upsample_log_uncertainty(
            log_uncertainty.float(),
            update["uncertainty_upsampling_features"].float(),
            stem_features.float(),
        )
        return {
            "disparity": disparity,
            "log_uncertainty": log_uncertainty,
            "heading_normal": heading_normal,
        }
