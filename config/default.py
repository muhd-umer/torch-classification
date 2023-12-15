"""
Default configuration file for Torch Classification.
"""

import os

from box import Box


def get_config():
    cfg = Box()

    # Set root directory
    cfg.root_dir = os.path.abspath(".")

    # Misc
    cfg.seed = 42

    # Dataset
    cfg.data_dir = os.path.abspath(os.path.join(cfg.root_dir, "data"))
    cfg.batch_size = 4
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.num_classes = 100
    cfg.val_size = 0.1
    cfg.img_size = 224  # desired image size, not actual image size
    cfg.mean = [0.5071, 0.4867, 0.4408]
    cfg.std = [0.2675, 0.2565, 0.2761]
    cfg.transform_set = "default"

    # Model
    cfg.model_name = "efficientnetv2_rw_m"  # must be in timm.list_models()
    cfg.pretrained = True

    # Training
    cfg.model_dir = os.path.abspath(os.path.join(cfg.root_dir, "weights"))
    cfg.num_epochs = 50
    cfg.lr = 0.001

    return cfg
