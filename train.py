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
import os
import warnings

import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from termcolor import colored
from torchinfo import summary

from config import get_config
from data import get_cifar100_loaders
from models import (
    EfficientNetV2,
    ImageClassifier,
    MBConv,
    MBConvConfig,
    efficientnet_v2_init,
    get_structure,
)
from utils import EMACallback, SimplifiedProgressBar, get_transforms

# Common setup
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
plt.rcParams["font.family"] = "STIXGeneral"


def train(
    mode,
    cfg,
    accelerator,
    devices,
    rich_progress,
    test_mode=False,
    resume=False,
    weights=None,
    logger_backend="tensorboard",
):
    if logger_backend == "tensorboard":
        logger = pl.pytorch.loggers.TensorBoardLogger(save_dir=cfg.log_dir, name=".")

    elif logger_backend == "wandb":
        logger = pl.pytorch.loggers.WandbLogger(
            project="torch-classification", save_dir=cfg.log_dir
        )
    else:
        raise ValueError(
            colored(
                "Provide a valid logger (tensorboard, wandb)",
                "red",
            )
        )

    # Instantiate
    train_transform, test_transform = get_transforms(cfg)
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        steps_per_epoch,
    ) = get_cifar100_loaders(
        cfg.data_dir,
        train_transform,
        test_transform,
        cfg.batch_size,
        cfg.num_workers,
        val_size=0.1,
        dataset_type=cfg.dataset_type,
        return_steps=True,
    )

    # Divide steps per epoch by number of GPUs
    if devices != "auto":
        steps_per_epoch = steps_per_epoch // devices

    cfg.steps_per_epoch = steps_per_epoch

    if mode == "finetune":
        # Create the model
        model = timm.create_model(
            cfg.model_name_timm, pretrained=True, num_classes=cfg.num_classes
        )

    elif mode == "train":
        # Create the model
        residual_config = [
            MBConvConfig(*layer_config)
            for layer_config in get_structure(cfg.model_name)
        ]
        model = EfficientNetV2(
            residual_config,
            1280,
            cfg.num_classes,
            dropout=0.1,
            stochastic_depth=0.2,
            block=MBConv,
            act_layer=nn.SiLU,
        )
        efficientnet_v2_init(model)
    else:
        raise ValueError(
            colored(
                "Provide a valid mode (train, finetune)",
                "red",
            )
        )

    if os.getenv("LOCAL_RANK", "0") == "0":
        yaml_cfg = cfg.to_yaml()

        os.makedirs(cfg.log_dir, exist_ok=True)

        print(colored(f"Config:", "green", attrs=["bold"]))
        print(colored(yaml_cfg))
        model_title = cfg.model_name_timm if mode == "finetune" else cfg.model_name
        print(colored(f"Model: {model_title}", "green", attrs=["bold"]))
        summary(
            model,
            input_size=(3, cfg.img_size, cfg.img_size),
            depth=1,
            batch_dim=0,
            device="cpu",
        )

    model = ImageClassifier(model, cfg)

    # Load from checkpoint if weights are provided
    if weights is not None:
        model.load_state_dict(torch.load(weights)["state_dict"])

    if logger_backend == "wandb":
        logger.watch(model, log="all", log_freq=100)

    # Create a PyTorch Lightning trainer with the required callbacks
    if rich_progress:
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
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=16,
            max_epochs=cfg.num_epochs,
            enable_model_summary=False,
            check_val_every_n_epoch=5,
            logger=logger,
            callbacks=[
                # pl_callbacks.RichModelSummary(max_depth=3),
                pl_callbacks.RichProgressBar(theme=theme),
                pl_callbacks.ModelCheckpoint(
                    dirpath=cfg.model_dir,
                    filename=f"{cfg.model_name}_best_model",
                ),
                EMACallback(decay=0.999),
                pl_callbacks.LearningRateMonitor(logging_interval="step"),
            ],
        )
    else:
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=16,
            max_epochs=cfg.num_epochs,
            enable_model_summary=False,
            check_val_every_n_epoch=5,
            logger=logger,
            callbacks=[
                # pl_callbacks.ModelSummary(max_depth=3),
                SimplifiedProgressBar(),
                pl_callbacks.ModelCheckpoint(
                    dirpath=cfg.model_dir,
                    filename=f"{cfg.model_name}_best_model",
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
    parser.add_argument("--mode", type=str, help="Training mode (train, finetune)")
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
        "--dataset-type",
        type=str,
        default=cfg.dataset_type,
        help="Dataset type (default, imagefolder)",
    )
    parser.add_argument(
        "--transform-set",
        type=str,
        default=cfg.transform_set,
        help="Transform set (imagenet, cifar, svhn, default)",
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
    parser.add_argument(
        "--logger-backend",
        type=str,
        default="tensorboard",
        help="Logger backend (tensorboard, wandb)",
    )
    args = parser.parse_args()

    cfg.update(args.__dict__)

    if cfg.transform_set not in ["imagenet", "cifar", "svhn", "default"]:
        raise ValueError(
            colored(
                "Provide a valid transform set (imagenet, cifar, svhn, default)",
                "red",
            )
        )

    if cfg.dataset_type == "default":
        cfg.mean = [0.5071, 0.4867, 0.4408]
        cfg.std = [0.2675, 0.2565, 0.2761]
    elif cfg.dataset_type == "imagefolder":
        cfg.mean = [0.5081, 0.4843, 0.4414]
        cfg.std = [0.2888, 0.2726, 0.2962]
    else:
        raise ValueError(
            colored(
                "Provide a valid dataset type (default, imagefolder)",
                "red",
            )
        )

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
        args.mode,
        cfg,
        args.accelerator,
        args.devices,
        args.rich_progress,
        args.test_only,
        args.resume,
        args.weights if args.resume or args.test_only else None,
        args.logger_backend,
    )
