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


class ImageClassifier(pl.LightningModule):
    """
    Image Classifier class that extends PyTorch Lightning Module.
    """

    def __init__(self, model: nn.Module, cfg: dict):
        """
        Initialize the ImageClassifier.

        Args:
            model (nn.Module): The model to use for classification.
            cfg (dict): Configuration dictionary.
        """
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Model output tensor.
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Training step for each batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The current batch of input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            The loss for this step.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Validation step for each batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The current batch of input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            The loss for this step.
        """
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
        """
        Test step for each batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The current batch of input data and labels.
            batch_idx (int): The index of the current batch.
        """
        x, y = batch
        y_hat = self(x)

        # calculate accuracy
        _, preds = torch.max(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(
            preds, y, num_classes=100, task="multiclass"
        )
        self.log("test_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Configure the optimizers for training.

        Returns:
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = optim.RAdam(self.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.num_epochs, eta_min=0
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
