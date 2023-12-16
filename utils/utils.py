import sys

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from tqdm import tqdm


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


class SimplifiedProgressBar(TQDMProgressBar):
    """
    Simplified progress bar for non-interactive terminals.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


def get_mean_std(loader: DataLoader):
    """
    Calculate the mean and standard deviation of the data in the loader.

    Args:
        loader (DataLoader): The DataLoader for which the mean and std are calculated.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation of the data in the loader.
    """
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

    return mean, std
