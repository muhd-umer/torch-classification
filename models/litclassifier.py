"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Image classification model
"""

from typing import Tuple

import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics
from torch import nn, optim


class ImageClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

        # calculate accuracy
        _, preds = torch.max(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(
            preds, y, num_classes=100, task="multiclass"
        )
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)

        # calculate accuracy
        _, preds = torch.max(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(
            preds, y, num_classes=100, task="multiclass"
        )
        self.log("test_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr,
            epochs=self.cfg.num_epochs,
            steps_per_epoch=self.cfg.steps_per_epoch,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
