"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Architecture of the model
"""

import copy
from collections import OrderedDict

import torch.nn as nn

from .blocks import ConvBNAct, MBConv


class EfficientNetV2(nn.Module):
    """
    Pytorch Implementation of EfficientNetV2

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
        act_layer=nn.Mish,
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
                        nn.Sequential(
                            nn.Linear(out_channels, out_channels // 2),
                            nn.ReLU(),
                            nn.Linear(out_channels // 2, num_classes),
                        )
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
    """
    Initialize the weights and biases of the layers in the given model.

    For Conv2d layers, weights are initialized using Kaiming Normal initialization,
    and biases are initialized to zero.

    For BatchNorm2d and GroupNorm layers, weights are initialized to one,
    and biases are initialized to zero.

    For Linear layers, weights are initialized from a normal distribution with mean 0 and std 0.01,
    and biases are initialized to zero.

    Args:
        model (torch.nn.Module): The model whose layers are to be initialized.
    """
    for m in model.modules():
        # Conv2d
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # BatchNorm2d and GroupNorm
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Linear
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
