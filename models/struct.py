"""
Structure of EfficientNetV2 models.

Source:
https://github.com/hankyul2/EfficientNetV2-pytorch/

Contains the structure of EfficientNetV2 models.

The structure is a list of tuples, where each tuple represents a block.

The tuple contains the following values:
- e: expand_ratio
- k: kernel_size
- s: stride
- in: input_channels
- out: output_channels
- xN: num_repeat
- se: use_se
- fused: fused_conv
"""

import torch.nn as nn

efficientnet_v2_structures = {
    "efficientnet_v2_s": [
        # e k  s  in  out xN  se   fused
        (1, 3, 1, 24, 24, 2, False, True),
        (4, 3, 2, 24, 48, 4, False, True),
        (4, 3, 2, 48, 64, 4, False, True),
        (4, 3, 2, 64, 128, 6, True, False),
        (6, 3, 1, 128, 160, 9, True, False),
        (6, 3, 2, 160, 256, 15, True, False),
    ],
    "efficientnet_v2_m": [
        # e k  s  in  out xN  se   fused
        (1, 3, 1, 24, 24, 3, False, True),
        (4, 3, 2, 24, 48, 5, False, True),
        (4, 3, 2, 48, 80, 5, False, True),
        (4, 3, 2, 80, 160, 7, True, False),
        (6, 3, 1, 160, 176, 14, True, False),
        (6, 3, 2, 176, 304, 18, True, False),
        (6, 3, 1, 304, 512, 5, True, False),
    ],
    "efficientnet_v2_l": [
        # e k  s  in  out xN  se   fused
        (1, 3, 1, 32, 32, 4, False, True),
        (4, 3, 2, 32, 64, 7, False, True),
        (4, 3, 2, 64, 96, 7, False, True),
        (4, 3, 2, 96, 192, 10, True, False),
        (6, 3, 1, 192, 224, 19, True, False),
        (6, 3, 2, 224, 384, 25, True, False),
        (6, 3, 1, 384, 640, 7, True, False),
    ],
    "efficientnet_v2_xl": [
        # e k  s  in  out xN  se   fused
        (1, 3, 1, 32, 32, 4, False, True),
        (4, 3, 2, 32, 64, 8, False, True),
        (4, 3, 2, 64, 96, 8, False, True),
        (4, 3, 2, 96, 192, 16, True, False),
        (6, 3, 1, 192, 256, 24, True, False),
        (6, 3, 2, 256, 512, 32, True, False),
        (6, 3, 1, 512, 640, 8, True, False),
    ],
}


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
        act (nn.Module, optional): Activation function. Defaults to nn.Mish.
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
        act=nn.Mish,
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


def get_structure(model_name):
    """
    Get the structure of the EfficientNetV2 model.

    Args:
        model_name (str): Name of the model.

    Returns:
        structure (list): Structure of the model.
    """
    structure = efficientnet_v2_structures.get(model_name, None)
    if structure is None:
        raise ValueError(f"Invalid model name: {model_name}")

    return structure
