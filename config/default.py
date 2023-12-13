"""
Default configuration file for Torch Classification.
"""

import os

import ml_collections


def get_cifar100_config():
    cfg = ml_collections.ConfigDict()

    # Misc
    cfg.seed = 42

    # Dataset
    cfg.data_root = os.path.abspath("../data/")
    cfg.batch_size = 64
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.num_classes = 100
    cfg.val_size = 0.1

    # Training
    cfg.model_dir = os.path.abspath("../weights/")

    return cfg
