"""Pytorch models"""
# pylint: disable=import-error
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import F1Score

LOSSES = {
    "classification": F.binary_cross_entropy_with_logits,
    "regression": F.mse_loss,
}


def get_train_val_sizes(data: TensorDataset, ratio: float) -> Tuple[int, int]:
    """
    Compute number of samples for train and validation loaders from
    a tensor dataset
    """
    data_size = int(len(data))
    train_size = int(data_size * ratio)
    val_size = data_size - train_size

    return train_size, val_size


# pylint: disable=too-many-arguments
def create_loaders(
    data: TensorDataset,
    ratio: float,
    batch_size: int,
    num_workers: int,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders with desired train/validation split ratio
    """
    logger.info(
        f"Creating dataloaders with {ratio*100:.0f}/{(1 - ratio)*100:.0f}"
        "train/test split \U0001F52A"
    )
    train_size, val_size = get_train_val_sizes(data, ratio)

    train_set, val_set = torch.utils.data.random_split(
        data, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_set,
        shuffle=shuffle_train,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=shuffle_val,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info("Successfully created train and validation loader \U0001F917")
    return train_loader, val_loader


# pylint: disable=invalid-name
# pylint: disable=too-few-public-methods
class Encoder(nn.Module):
    """
    Simple encoder
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Setup model layers"""
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        """Forward pass through layers"""
        return self.layers(x)


class Linear_Encoder(nn.Module):
    """
    Simple encoder without nonlinear activation functions
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Setup model layers"""
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 10), nn.Linear(10, self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        """Forward pass through layers"""
        return self.layers(x)


class SimpleDnn(pl.LightningModule):
    """
    Simple dense DNN with train and validation loop
    """

    def __init__(self, encoder: nn.Module, task_type: str) -> None:
        """Setup model layers"""
        super().__init__()
        self.encoder = encoder
        self.f1 = F1Score(task="binary", num_classes=self.encoder.output_dim)
        self.loss = LOSSES[task_type]

    # pylint: disable=unused-argument
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Perform training step"""
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.encoder(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss.item())
        return loss

    # pylint: disable=unused-argument
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform validation step"""
        x, y = batch
        y_hat = self.encoder(x)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss.item())
        # self.log_dict({"val_loss": val_loss.item()})

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """Perform validation step"""
        x, _ = batch
        y_hat = F.softmax(self.encoder(x), dim=0)
        return y_hat

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
