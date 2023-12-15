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
