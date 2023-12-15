"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Architecture of the model
"""


import torch
import torch.nn as nn
from torch import nn


class EfficientNetV2(nn.Module):
    """
    EfficientNetV2 architecture

    Reference:
    [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
    """

    def __init__(self):
        super(EfficientNetV2, self).__init__()
        raise NotImplementedError


if __name__ == "__main__":
    # Test
    model = EfficientNetV2()
    print(model)
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)
