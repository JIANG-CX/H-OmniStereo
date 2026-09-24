"""Fixed recurrent disparity and uncertainty refinement blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from homnistereo.model.submodule import EdgeNextConvEncoder


class DisparityHead(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
            EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class BasicMotionEncoder(nn.Module):
    def __init__(self, config, correlation_groups: int = 28) -> None:
        super().__init__()
        correlation_channels = (
            config.corr_levels * (2 * config.corr_radius + 1) * (correlation_groups + 1)
        )
        self.convc1 = nn.Conv2d(correlation_channels, 256, kernel_size=1)
        self.convc2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, kernel_size=7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(320, 127, kernel_size=3, padding=1)

    def forward(
        self, disparity: torch.Tensor, correlation: torch.Tensor
    ) -> torch.Tensor:
        correlation = F.relu(self.convc1(correlation))
        correlation = F.relu(self.convc2(correlation))
        disparity_features = F.relu(self.convd1(disparity))
        disparity_features = F.relu(self.convd2(disparity_features))
        features = F.relu(
            self.conv(torch.cat([correlation, disparity_features], dim=1))
        )
        return torch.cat([features, disparity], dim=1)


def pool2x(inputs: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(inputs, kernel_size=3, stride=2, padding=1)


def interpolate_like(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        inputs, reference.shape[2:], mode="bilinear", align_corners=True
    )


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int) -> None:
        super().__init__()
        channels = hidden_dim + input_dim
        padding = kernel_size // 2
        self.convz = nn.Conv2d(channels, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(channels, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(channels, hidden_dim, kernel_size, padding=padding)

    def forward(self, hidden, inputs, hidden_inputs):
        update = torch.sigmoid(self.convz(hidden_inputs))
        reset = torch.sigmoid(self.convr(hidden_inputs))
        candidate = torch.tanh(self.convq(torch.cat([reset * hidden, inputs], dim=1)))
        return (1 - update) * hidden + update * candidate


class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 256) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_dim + hidden_dim, input_dim + hidden_dim, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, kernel_size=1)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, kernel_size=3)

    def forward(self, attention, hidden, *inputs):
        inputs = self.conv0(torch.cat(inputs, dim=1))
        hidden_inputs = self.conv1(torch.cat([inputs, hidden], dim=1))
        small = self.small_gru(hidden, inputs, hidden_inputs)
        large = self.large_gru(hidden, inputs, hidden_inputs)
        return small * attention + large * (1 - attention)


class RecurrentUpdateBlock(nn.Module):
    """The single three-level GRU topology used by the release checkpoint."""

    def __init__(self, config, hidden_dim: int = 128, volume_dim: int = 28) -> None:
        super().__init__()
        self.encoder = BasicMotionEncoder(config, volume_dim)
        self.gru16 = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        self.gru08 = SelectiveConvGRU(hidden_dim, hidden_dim * 3)
        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim * 3)
        self.disp_head = DisparityHead(hidden_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cov_gru = SelectiveConvGRU(hidden_dim, hidden_dim * 3)
        self.cov_head = DisparityHead(hidden_dim)
        self.cov_mask = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, hidden, context, correlation, disparity, attention, uncertainty_hidden
    ):
        hidden[2] = self.gru16(attention[2], hidden[2], context[2], pool2x(hidden[1]))
        hidden[1] = self.gru08(
            attention[1],
            hidden[1],
            context[1],
            pool2x(hidden[0]),
            interpolate_like(hidden[2], hidden[1]),
        )
        motion = torch.cat([context[0], self.encoder(disparity, correlation)], dim=1)
        recurrent_inputs = (motion, interpolate_like(hidden[1], hidden[0]))
        hidden[0] = self.gru04(attention[0], hidden[0], *recurrent_inputs)
        uncertainty_hidden = self.cov_gru(
            attention[0], uncertainty_hidden, *recurrent_inputs
        )
        return {
            "hidden": hidden,
            "upsampling_features": 0.25 * self.mask(hidden[0]),
            "disparity_delta": self.disp_head(hidden[0]),
            "uncertainty_hidden": uncertainty_hidden,
            "log_uncertainty": self.cov_head(uncertainty_hidden),
            "uncertainty_upsampling_features": 0.25 * self.cov_mask(uncertainty_hidden),
        }
