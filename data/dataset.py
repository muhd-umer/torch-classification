"""
Dataset class for CIFAR100.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torchvision import transforms


class CIFAR100(Dataset):
    """
    Dataset class for CIFAR100.
    """

    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (string): Root directory of dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version.
        """
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        # load all images into memory
        self.data = []
        self.targets = []
        if self.train:
            for i in range(1, 6):
                file = os.path.join(self.root, "train_batch_{}".format(i))
                with open(file, "rb") as f:
                    entry = np.load(f, encoding="latin1", allow_pickle=True)
                    self.data.append(entry["data"])
                    self.targets.extend(entry["fine_labels"])
        else:
            file = os.path.join(self.root, "test_batch")
            with open(file, "rb") as f:
                entry = np.load(f, encoding="latin1", allow_pickle=True)
                self.data.append(entry["data"])
                self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index.

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # convert to PIL image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        """
        Returns:
            int: Length of dataset.
        """
        return len(self.data)


def get_cifar100(
    root, train_transform=None, test_transform=None, val_size=0.1, shuffle=True
):
    """
    Get CIFAR100 dataset.

    Args:
        root (string): Root directory of dataset.
        train_transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        test_transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        val_size (float, optional): If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the validation split.
        shuffle (bool, optional): If True, the data will be split randomly.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = CIFAR100(root, train=True, transform=train_transform)
    test_dataset = CIFAR100(root, train=False, transform=test_transform)

    if val_size > 0.0:
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


def get_cifar100_transforms():
    """
    Get CIFAR100 transforms.

    Returns:
        tuple: (train_transform, test_transform)
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor()])

    return train_transform, test_transform


def get_cifar100_loaders(
    root,
    batch_size=128,
    num_workers=4,
    train_transform=None,
    test_transform=None,
    val_size=0.1,
    shuffle=True,
):
    """
    Get CIFAR100 loaders.
    Args:
        root (string): Root directory of dataset.
        batch_size (int, optional): How many samples per batch to load.
        num_workers (int, optional): How many subprocesses to use for data loading.
        train_transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        test_transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        val_size (float, optional): If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the validation split.
        shuffle (bool, optional): If True, the data will be split randomly.
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform, test_transform = get_cifar100_transforms()

    train_dataset, val_dataset, test_dataset = get_cifar100(
        root, train_transform, test_transform, val_size, shuffle
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
