"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Architecture of the model
"""

from functools import partial
from typing import Type

import torch
import torch.nn as nn
from torch import nn


class SEUnit(nn.Module):
    """
    This class represents a Squeeze-Excitation Unit.
    It is a type of attention mechanism that adaptively recalibrates channel-wise feature responses.

    Paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper

    Args:
        in_channel (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the hidden layer size. Defaults to 4.
        act1 (nn.Module, optional): First activation function. Defaults to nn.SiLU with inplace=True.
        act2 (nn.Module, optional): Second activation function. Defaults to nn.Sigmoid.
    """

    def __init__(
        self,
        in_channel: int,
        reduction_ratio: int = 4,
        act1: Type[nn.Module] = partial(nn.SiLU, inplace=True),
        act2: Type[nn.Module] = nn.Sigmoid,
    ):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.ModuleList(
            [
                nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True),
                nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True),
            ]
        )
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x: torch.Tensor):
        return x * self.act2(self.fc[1](self.act1(self.fc[0](self.avg_pool(x)))))


class ConvBNAct(nn.Sequential):
    """
    This class represents a Convolution-Normalization-Activation Module.
    It is a sequence of convolution, normalization, and activation operations.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int,
        groups: int,
        norm_layer: Type[nn.Module],
        act: Type[nn.Module],
        conv_layer: Type[nn.Module] = nn.Conv2d,
    ):
        """
        Initialize the ConvBNAct module. It is a sequence of convolution, normalization, and activation operations.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            groups (int): Number of blocked connections from input channels to output channels.
            norm_layer (nn.Module): Normalization layer.
            act (nn.Module): Activation function.
            conv_layer (nn.Module, optional): Convolution layer. Defaults to nn.Conv2d.
        """
        super(ConvBNAct, self).__init__(
            conv_layer(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channel),
            act(),
        )


class StochasticDepth(nn.Module):
    """
    This class represents a Stochastic Depth module.
    It randomly drops layers during training to reduce overfitting.

    Paper: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39
    """

    def __init__(self, prob: float, mode: str):
        """
        Initialize the Stochastic Depth module. It randomly drops layers during training to reduce overfitting.

        Args:
            prob (float): Probability of a layer to be dropped.
            mode (str): "row" or "col", determines the shape of the Bernoulli distribution tensor.
        """
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x: torch.Tensor):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == "row" else [1]
            return x * torch.bernoulli(
                torch.empty(shape, device=x.device), self.survival
            ).div_(self.survival)
