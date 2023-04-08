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

from loguru import logger
from data_generation import MEANS, VARIANCES, KEY, generate_mv_data, NUM_SAMPLES
from torch.utils.data import DataLoader
import torch
from torch.utils.data import TensorDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch.nn.functional as F
from typing import Tuple
from sklearn.datasets import make_circles

BATCH_SIZE = 32

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
    subset_broken_train: bool = False
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
    if subset_broken_train:
        train_set = TensorDataset(train_set[0][0][0].repeat(train_size).reshape(-1,2), 
                                  train_set[1][0][0].repeat(train_size).reshape(-1, 2))
    
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
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 10),
            nn.Linear(10, self.output_dim),
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


class LargeEncoder(nn.Module):
    """
    Simple encoder
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Setup model layers"""
        super().__init__()
        self.dense1 =  nn.Linear(input_dim, 10)
        self.dense2 = nn.Linear(10, 10)
        self.dense3 = nn.Linear(10, 10)
        self.dense4 = nn.Linear(10, 10)
        self.dense5 = nn.Linear(10, 10)
        self.dense6 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()
        
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        """Forward pass through layers"""
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        x = self.relu(x)
        x = self.dense5(x)
        x = self.relu(x)
        x = self.dense6(x)
        return x

class LargeLinearEncoder(nn.Module):
    """
    Simple encoder
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Setup model layers"""
        super().__init__()
        self.dense1 =  nn.Linear(input_dim, 10)
        self.dense2 = nn.Linear(10, 10)
        self.dense3 = nn.Linear(10, 10)
        self.dense4 = nn.Linear(10, 10)
        self.dense5 = nn.Linear(10, 10)
        self.dense6 = nn.Linear(10, output_dim)
        
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        """Forward pass through layers"""
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return x


def generate_linear_dataloaders() -> Tuple[DataLoader]:
    # generate data
    data, labels = generate_mv_data(KEY, MEANS, VARIANCES, NUM_SAMPLES, 3)

    # format labels and set up dataloaders
    labels_one_hot = F.one_hot(torch.Tensor(np.hstack((np.array(labels)))).to(torch.int64), num_classes=3).float() 
    data_linear = TensorDataset(
                    torch.Tensor(np.vstack((np.array(data)))), labels_one_hot
                )
    train_loader_linear, val_loader_linear = create_loaders(data = data_linear, 
                                                            ratio = 0.8, 
                                                            num_workers = 0, 
                                                            shuffle_train = True, 
                                                            shuffle_val = False, 
                                                            batch_size = 32)

    return train_loader_linear, val_loader_linear

def generate_nonlinear_dataloaders(break_loader: bool = False) -> Tuple[DataLoader]:
    data_circles, label_circles = make_circles(n_samples=NUM_SAMPLES, factor=0.5, noise=0.05)
    # concentric circles
    data_tensor_circles = TensorDataset(
                    torch.Tensor(data_circles), F.one_hot(torch.Tensor(label_circles).to(torch.int64), num_classes=2).float() 
                )
    print(break_loader)
    train_loader_circles, val_loader_circles = create_loaders(data = data_tensor_circles, 
                                                              ratio = 0.8,  
                                                              batch_size = 32, 
                                                              num_workers = 0,
                                                              shuffle_train = True, 
                                                              shuffle_val = False, 
                                                              subset_broken_train = break_loader)

    return train_loader_circles, val_loader_circles


def train_model(train_loader: DataLoader, 
                val_loader: DataLoader, 
                save_dir_logger: str = 'metrics_csv', 
                name_logger: str = 'linear', 
                break_activations: str = False, 
                freeze_weights: str = False, 
                freeze_bias: str = False, 
                input_dim: int = 2, 
                output_dim: int = 3,
                weight_watcher: bool = False) -> None:
    
    logger.info('Define model and logger')

    if not weight_watcher:
        simple_dnn = SimpleDnn(Encoder(input_dim = input_dim, output_dim = output_dim), task_type = 'classification')
    else:
        simple_dnn = SimpleDnn(LargeEncoder(input_dim = input_dim, output_dim = output_dim), task_type = 'classification')

    if break_activations:
        logger.info('CAUTION: training with linear activation functions')
        if not weight_watcher:
            simple_dnn = SimpleDnn(Linear_Encoder(input_dim = input_dim, 
                                                output_dim = output_dim), 
                                task_type = 'classification')
        else:
            simple_dnn = SimpleDnn(LargeLinearEncoder(input_dim = input_dim, 
                                                output_dim = output_dim), 
                                task_type = 'classification')
    elif freeze_weights:
        num_params = len([param for param in simple_dnn.parameters()])
        for i, param in zip(range(num_params), simple_dnn.parameters()):
            if i in [num_params-1, num_params -2] :
                param.requires_grad = False
    
    elif freeze_bias:
        num_params = len([param for param in simple_dnn.parameters()])
        for i, param in zip(range(num_params), simple_dnn.parameters()):
            if i == num_params-1 :
                param.requires_grad = False
    
    csv_logger = CSVLogger(save_dir=save_dir_logger, name = name_logger)

    logger.info(f'Saving logs to {save_dir_logger}/{name_logger}')
    
    logger.info('Start model training')
    trainer = pl.Trainer(logger=csv_logger, max_epochs = 50, log_every_n_steps=20)
    trainer.fit(model=simple_dnn, train_dataloaders = train_loader, val_dataloaders = val_loader)
    logger.info('Training finished successfully')