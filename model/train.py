# Standard Imports
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "..", "."))  # add parent dir to path

import argparse
from typing import Tuple

import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics
from torch import nn, optim

torch.set_float32_matmul_precision("medium")
# Custom imports
from config import *
from data import *

# Use STIX font for math plotting
plt.rcParams["font.family"] = "STIXGeneral"

import warnings

import torchvision
from termcolor import colored
from torchvision import transforms

warnings.filterwarnings("ignore")


def train(cfg, accelerator, devices, rich_progress):
    # Training
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    train_dataloader, val_dataloader, test_dataloader = get_cifar100_loaders(
        cfg.data_dir,
        cfg.batch_size,
        cfg.num_workers,
        train_transform,
        test_transform,
        val_size=0.1,
    )

    class ImageClassifier(pl.LightningModule):
        def __init__(self, model: nn.Module, cfg: dict):
            super().__init__()
            self.model = model
            self.cfg = cfg
            self.loss = nn.CrossEntropyLoss()

        def forward(self, x: torch.Tensor):
            return self.model(x)

        def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            self.log("val_loss", loss)

            # calculate accuracy
            _, preds = torch.max(y_hat, dim=1)
            acc = torchmetrics.functional.accuracy(
                preds, y, num_classes=100, task="multiclass"
            )
            self.log("val_acc", acc)

            return loss

        def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)

            # calculate accuracy
            _, preds = torch.max(y_hat, dim=1)
            acc = torchmetrics.functional.accuracy(
                preds, y, num_classes=100, task="multiclass"
            )
            self.log("test_loss", loss)
            self.log("test_acc", acc)

            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            return [optimizer], [scheduler]

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
    model = timm.create_model("resnet50", pretrained=True, num_classes=100)
    model = ImageClassifier(model, cfg)

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
            ],
        )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Evaluate the model on the test set
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    cfg = get_cifar100_config()

    # Add argument parsing with cfg overrides
    parser = argparse.ArgumentParser(description="Train a model on CIFAR100 dataset")
    parser.add_argument("--data-dir", type=str, default=cfg.data_dir)
    parser.add_argument("--model-dir", type=str, default=cfg.model_dir)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--num-epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--rich-progress", action="store_true")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    args = parser.parse_args()

    cfg.update(args.__dict__)

    print(colored(f"Config:", "green"))
    print(cfg)

    # Train the model
    if args.devices != "auto":
        args.devices = int(args.devices)
    train(cfg, args.accelerator, args.devices, args.rich_progress)
