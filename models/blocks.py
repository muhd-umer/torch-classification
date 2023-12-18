"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Architecture of the model
"""

from collections import OrderedDict
from functools import partial
from typing import Type

import torch
import torch.nn as nn


class SEUnit(nn.Module):
    """
    This class represents a Squeeze-Excitation Unit.
    It is a type of attention mechanism that adaptively recalibrates channel-wise feature responses.

    Paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper

    Args:
        in_channel (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the hidden layer size. Defaults to 4.
        act1 (nn.Module, optional): First activation function. Defaults to nn.Mish with inplace=True.
        act2 (nn.Module, optional): Second activation function. Defaults to nn.Sigmoid.
    """

    def __init__(
        self,
        in_channel: int,
        reduction_ratio: int = 4,
        act1: Type[nn.Module] = partial(nn.Mish, inplace=True),
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


class MBConv(nn.Module):
    """
    This class represents the main building blocks of EfficientNet.

    Args:
        c (MBConvConfig): Configuration of the MBConv block.
        sd_prob (float, optional): Stochastic depth probability. Defaults to 0.0.
    """

    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []

        # fmt: off
        if c.expand_ratio == 1 or c.fused:
            block.append(("fused", ConvBNAct(c.in_ch, inter_channel, c.kernel,
                                             c.stride, 1, c.norm_layer, c.act)))
            if c.fused:
                block.append(("fused_point_wise",
                              ConvBNAct(inter_channel, c.out_ch,
                                        1, 1, 1, c.norm_layer, nn.Identity)))
        else:
            block.extend([
                ("linear_bottleneck", ConvBNAct(c.in_ch, inter_channel,
                                                1, 1, 1, c.norm_layer, c.act)),
                ("depth_wise", ConvBNAct(inter_channel, inter_channel,
                                         c.kernel, c.stride, inter_channel,
                                         c.norm_layer, c.act)),
                ("se", SEUnit(inter_channel, 4 * c.expand_ratio)),
                ("point_wise", ConvBNAct(inter_channel, c.out_ch,
                                         1, 1, 1, c.norm_layer, nn.Identity))
            ])
        # fmt: on

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out
