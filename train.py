"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Training script for Pytorch models [Pytorch Lightning]

Usage:
    >>> python train.py --help
    >>> python train.py --data-dir <path> --model-dir <path> --batch-size <int> --num-workers <int> --num-epochs <int> --lr <float> --rich-progress --accelerator <str> --devices <str> --weights <path> --resume --test-only
    
Example:
    Use the default config:
    >>> python train.py
    
    Override the config:
    >>> python train.py --data-dir data --model-dir models --batch-size 128 --num-workers 8 --num-epochs 100 --lr 0.001 --rich-progress --accelerator gpu --devices 1 --weights models/best_model.ckpt --resume --test-only
"""

import argparse
import warnings

import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks
import matplotlib.pyplot as plt
import torch
from termcolor import colored

from config import *
from data import *
from models import EfficientNetV2, ImageClassifier
from utils import *

# Common setup
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
plt.rcParams["font.family"] = "STIXGeneral"


def train(
    cfg,
    accelerator,
    devices,
    rich_progress,
    test_mode=False,
    resume=False,
    weights=None,
):
    train_transform, test_transform = get_transforms(cfg)
    train_dataloader, val_dataloader, test_dataloader = get_cifar100_loaders(
        cfg.data_dir,
        train_transform,
        test_transform,
        cfg.batch_size,
        cfg.num_workers,
        val_size=0.1,
    )

    theme = pl_callbacks.progress.rich_progress.RichProgressBarTheme(
        description="black",
        progress_bar="cyan",
        progress_bar_finished="green",
        progress_bar_pulse="#6206E0",
        batch_progress="cyan",
        time="grey82",
        processing_speed="grey82",
        metrics="black",
    )

    # Create the model
    model = EfficientNetV2(num_classes=cfg.num_classes)
    model = ImageClassifier(model, cfg)

    # Load from checkpoint if weights are provided
    if weights is not None:
        model.load_state_dict(torch.load(weights)["state_dict"])

    # Create a PyTorch Lightning trainer with the required callbacks
    if rich_progress:
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=cfg.num_epochs,
            enable_model_summary=False,
            callbacks=[
                pl_callbacks.RichModelSummary(max_depth=3),
                pl_callbacks.RichProgressBar(theme=theme),
                pl_callbacks.ModelCheckpoint(
                    dirpath=cfg.model_dir,
                    filename="best_model",
                ),
                EMACallback(decay=0.999),
                pl_callbacks.LearningRateMonitor(logging_interval="step"),
            ],
        )
    else:
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=cfg.num_epochs,
            enable_model_summary=False,
            callbacks=[
                pl_callbacks.ModelSummary(max_depth=3),
                pl_callbacks.ModelCheckpoint(
                    dirpath=cfg.model_dir,
                    filename="best_model",
                ),
                EMACallback(decay=0.999),
                pl_callbacks.LearningRateMonitor(logging_interval="step"),
            ],
        )

    # Train the model
    if not test_mode:
        if resume:
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=weights)
        trainer.fit(model, train_dataloader, val_dataloader)

    # Evaluate the model on the test set
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    cfg = get_config()

    # Add argument parsing with cfg overrides
    parser = argparse.ArgumentParser(description="Train a model on CIFAR100 dataset")
    parser.add_argument(
        "--data-dir", type=str, default=cfg.data_dir, help="Directory for the data"
    )
    parser.add_argument(
        "--model-dir", type=str, default=cfg.model_dir, help="Directory for the model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=cfg.batch_size, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=cfg.num_workers,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=cfg.num_epochs,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--lr", type=float, default=cfg.lr, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--rich-progress", action="store_true", help="Use rich progress bar"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, gpu, tpu, etc.)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Devices to use for training (auto, cpu, gpu, etc.)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the weights file for the model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the provided weights",
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only test the model, do not train"
    )
    args = parser.parse_args()

    cfg.update(args.__dict__)
    yaml_cfg = cfg.to_yaml()

    print(colored(f"Config:", "green", attrs=["bold"]))
    print(colored(yaml_cfg, "white"))

    # Train the model
    if args.devices != "auto":
        args.devices = int(args.devices)
    if (args.resume or args.test_only) and args.weights is None:
        raise ValueError(
            colored(
                "Provide the path to the weights file using --weights",
                "red",
            )
        )

    train(
        cfg,
        args.accelerator,
        args.devices,
        args.rich_progress,
        args.test_only,
        args.resume,
        args.weights if args.resume or args.test_only else None,
    )
