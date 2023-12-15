"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Architecture of the model
"""

import torch
import torch.nn as nn
from torch import nn


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        se_ratio: float,
        bn_momentum: float,
        bn_epsilon: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.drop_path_rate = kwargs.get("drop_path_rate", 0.0)
        self.expand = in_channels != int(out_channels * expand_ratio)
        mid_channels = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels, momentum=bn_momentum, eps=bn_epsilon),
            nn.SiLU(),
            # dw
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels, momentum=bn_momentum, eps=bn_epsilon),
            # se
            SqueezeExcite(mid_channels, int(in_channels * se_ratio)),
            nn.SiLU(),
            # pw-linear
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_epsilon),
        )

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_path_rate > 0.0:
            keep_prob = 1.0 - self.drop_path_rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(
                shape, dtype=x.dtype, device=x.device
            )
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
        else:
            output = x
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self._drop_path(self.conv(x))
        else:
            return self.conv(x)


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)
