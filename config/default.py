"""
Default configuration file for Torch Classification.
"""

import os

import torch
from box import Box


def get_config():
    cfg = Box()

    # Set root directories
    cfg.root_dir = os.path.abspath(".")
    cfg.log_dir = os.path.abspath(os.path.join(cfg.root_dir, "logs"))

    # Misc
    cfg.seed = 42

    # Dataset
    cfg.data_dir = os.path.abspath(os.path.join(cfg.root_dir, "data"))
    cfg.dataset_type = "default"  # default or imagefolder
    cfg.batch_size = 4
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.num_classes = 100
    cfg.val_size = 0.1
    cfg.img_size = 224  # desired image size, not actual image size
    cfg.transform_set = "default"  # imagenet, cifar, svhn, default

    # CIFAR-100 Original
    # Mean: tensor([0.5071, 0.4867, 0.4408])
    # Std: tensor([0.2675, 0.2565, 0.2761])

    # CIFAR-100 Upscaled
    # Mean: tensor([0.5081, 0.4843, 0.4414])
    # Std: tensor([0.2888, 0.2726, 0.2962])

    # Model
    cfg.model_name = "efficientnet_v2_m"
    cfg.model_name_timm = "tf_efficientnetv2_m.in21k"  # must be in list of timm models
    cfg.pretrained = True
    cfg.model_dir = os.path.abspath(os.path.join(cfg.root_dir, "weights"))

    # Training
    cfg.num_epochs = 100
    cfg.lr = 0.0005
    cfg.weight_decay = 0.005
    cfg.momentum = 0.9
    cfg.rho = 0.05
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg
