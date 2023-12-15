"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Architecture of the model
"""

import copy
from collections import OrderedDict
from functools import partial
from typing import Optional, Type

import torch
import torch.nn as nn
from torch import nn

from .structure import get_structure


class ConvBNAct(nn.Sequential):
    """
    This class represents a Convolution-Normalization-Activation Module.
    It is a sequence of convolution, normalization, and activation operations.

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


class StochasticDepth(nn.Module):
    """
    This class represents a Stochastic Depth module.
    It randomly drops layers during training to reduce overfitting.

    Paper: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39

    Args:
        prob (float): Probability of a layer to be dropped.
        mode (str): "row" or "col", determines the shape of the Bernoulli distribution tensor.
    """

    def __init__(self, prob: float, mode: str):
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


class MBConvConfig:
    """
    This class represents the configuration for an EfficientNet building block.

    Args:
        expand_ratio (float): Expansion ratio for the hidden layer size.
        kernel (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        layers (int): Number of layers.
        use_se (bool): Whether to use Squeeze-Excitation Unit.
        fused (bool): Whether to use fused convolution.
        act (nn.Module, optional): Activation function. Defaults to nn.SiLU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
    """

    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        in_ch: int,
        out_ch: int,
        layers: int,
        use_se: bool,
        fused: bool,
        act=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
    ):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(
            divisible, (int(new_channel + divisible / 2) // divisible) * divisible
        )
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


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


class EfficientNetV2(nn.Module):
    """Pytorch Implementation of EfficientNetV2

    Paper:
    https://arxiv.org/abs/2104.00298

    Args:
        layer_infos (list): list of MBConvConfig
        out_channels (int): output channel of the last layer
        num_classes (int): number of classes
        dropout (float): dropout rate
        stochastic_depth (float): stochastic depth rate
        block (nn.Module): building block
        act_layer (nn.Module): activation function
        norm_layer (nn.Module): normalization layer
    """

    def __init__(
        self,
        layer_infos,
        out_channels=1280,
        num_classes=0,
        dropout=0.2,
        stochastic_depth=0.0,
        block=MBConv,
        act_layer=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch
        self.out_channels = out_channels

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.stem = ConvBNAct(3, self.in_channel, 3, 2, 1, self.norm_layer, self.act)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))
        self.head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "bottleneck",
                        ConvBNAct(
                            self.final_stage_channel,
                            out_channels,
                            1,
                            1,
                            1,
                            self.norm_layer,
                            self.act,
                        ),
                    ),
                    ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten()),
                    ("dropout", nn.Dropout(p=dropout, inplace=True)),
                    (
                        "classifier",
                        nn.Linear(out_channels, num_classes)
                        if num_classes
                        else nn.Identity(),
                    ),
                ]
            )
        )

    def make_stages(self, layer_infos, block):
        return [
            layer
            for layer_info in layer_infos
            for layer in self.make_layers(copy.copy(layer_info), block)
        ]

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

    def change_dropout_rate(self, p):
        self.head[-2] = nn.Dropout(p=p, inplace=True)


def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


def get_efficientnet_v2(
    model_name, num_classes=0, dropout=0.1, stochastic_depth=0.2, **kwargs
):
    residual_config = [
        MBConvConfig(*layer_config) for layer_config in get_structure(model_name)
    ]
    model = EfficientNetV2(
        residual_config,
        1280,
        num_classes,
        dropout=dropout,
        stochastic_depth=stochastic_depth,
        block=MBConv,
        act_layer=nn.SiLU,
    )
    efficientnet_v2_init(model)

    return model
