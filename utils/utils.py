import numpy as np
import torch
import torchvision.transforms.v2 as v2


def numpy_collate(batch):
    """
    Collate function to use PyTorch datalaoders
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_transforms(cfg):
    """
    Get transforms for dataset.

    Returns:
        tuple: (train_transform, test_transform)
    """
    transform_dict = {
        "default": (
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(
                        (cfg.img_size, cfg.img_size),
                        interpolation=v2.InterpolationMode.BICUBIC,
                    ),
                    v2.RandomRotation(degrees=(-30, 30)),
                    v2.ColorJitter(
                        brightness=0.25, contrast=0, saturation=0.3, hue=0.2
                    ),
                    v2.RandomAdjustSharpness(sharpness_factor=1.75, p=0.25),
                    v2.RandomAutocontrast(p=0.25),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=cfg.mean, std=cfg.std),
                ]
            ),
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(
                        (cfg.img_size, cfg.img_size),
                        interpolation=v2.InterpolationMode.BICUBIC,
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=cfg.mean, std=cfg.std),
                ]
            ),
        ),
        "imagenet": (
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(
                        (cfg.img_size, cfg.img_size),
                        interpolation=v2.InterpolationMode.BICUBIC,
                    ),
                    v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=cfg.mean, std=cfg.std),
                ]
            ),
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(
                        (cfg.img_size, cfg.img_size),
                        interpolation=v2.InterpolationMode.BICUBIC,
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=cfg.mean, std=cfg.std),
                ]
            ),
        ),
        "cifar": (
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(
                        (cfg.img_size, cfg.img_size),
                        interpolation=v2.InterpolationMode.BICUBIC,
                    ),
                    v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=cfg.mean, std=cfg.std),
                ]
            ),
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(
                        (cfg.img_size, cfg.img_size),
                        interpolation=v2.InterpolationMode.BICUBIC,
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=cfg.mean, std=cfg.std),
                ]
            ),
        ),
        "svhn": (
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(
                        (cfg.img_size, cfg.img_size),
                        interpolation=v2.InterpolationMode.BICUBIC,
                    ),
                    v2.AutoAugment(v2.AutoAugmentPolicy.SVHN),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=cfg.mean, std=cfg.std),
                ]
            ),
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(
                        (cfg.img_size, cfg.img_size),
                        interpolation=v2.InterpolationMode.BICUBIC,
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=cfg.mean, std=cfg.std),
                ]
            ),
        ),
    }

    return transform_dict[cfg.transform_set]
