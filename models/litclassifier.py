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

from .sam import SAM


class ImageClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.automatic_optimization = False
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        optimizer = self.optimizers()

        def closure():
            loss = self.loss(self(x), y)
            loss.backward()
            return loss

        loss = self.loss(self(x), y)
        loss.backward()
        optimizer.step(closure)
        optimizer.zero_grad()

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
        base_optimizer = optim.SGD
        optimizer = SAM(
            self.model.parameters(),
            base_optimizer,
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
            rho=self.cfg.rho,
        )

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.num_epochs
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
