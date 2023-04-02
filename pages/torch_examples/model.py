"""Pytorch models"""
# pylint: disable=import-error
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import F1Score


# pylint: disable=invalid-name
# pylint: disable=too-few-public-methods
class Encoder(nn.Module):
    """
    Simple encoder
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Setup model layers"""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 10), nn.ReLU(), nn.Linear(10, output_dim)
        )
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        """Forward pass through layers"""
        return self.layers(x)


class SimpleDnn(pl.LightningModule):
    """
    Simple dense DNN with train and validation loop
    """

    def __init__(self, encoder: nn.Module) -> None:
        """Setup model layers"""
        super().__init__()
        self.encoder = encoder
        self.f1 = F1Score(task="binary", num_classes=self.encoder.output_dim)

    # pylint: disable=unused-argument
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Perform training step"""
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.encoder(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss.item())
        return loss

    # pylint: disable=unused-argument
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform validation step"""
        x, y = batch
        y_hat = self.encoder(x)
        val_loss = F.cross_entropy(y_hat, y)
        print(y_hat)
        val_f1 = self.f1(
            torch.round(torch.nn.functional.softmax(y_hat, dim=0)), y
        )
        # self.log("val_loss", val_loss.item())
        self.log_dict({"val_loss": val_loss.item(), "val_f1": val_f1})

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
