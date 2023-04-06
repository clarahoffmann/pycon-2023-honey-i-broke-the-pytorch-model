from loguru import logger
from data_generation import MEANS, VARIANCES, KEY, generate_mv_data, NUM_SAMPLES
from train_utils import SimpleDnn, Linear_Encoder,  Encoder, create_loaders
from torch.utils.data import DataLoader
import torch
from torch.utils.data import TensorDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch.nn.functional as F
import argparse
from typing import Tuple
from pathlib import Path
from csv_utils import format_csv
from sklearn.datasets import make_circles


BATCH_SIZE = 32


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
                output_dim: int = 3) -> None:
    
    logger.info('Define model and logger')
    if break_activations:
        logger.info('CAUTION: training with linear activation functions')
        simple_dnn = SimpleDnn(Linear_Encoder(input_dim = input_dim, 
                                              output_dim = output_dim), 
                               task_type = 'classification')
    elif freeze_weights:
        simple_dnn = SimpleDnn(Encoder(input_dim = input_dim, output_dim = output_dim), task_type = 'classification')

        num_params = len([param for param in simple_dnn.parameters()])
        for i, param in zip(range(num_params), simple_dnn.parameters()):
            if i in [num_params-1, num_params -2] :
                param.requires_grad = False
    
    elif freeze_bias:
        simple_dnn = SimpleDnn(Encoder(input_dim = input_dim, output_dim = output_dim), task_type = 'classification')

        num_params = len([param for param in simple_dnn.parameters()])
        for i, param in zip(range(num_params), simple_dnn.parameters()):
            if i == num_params-1 :
                param.requires_grad = False

    else:
        simple_dnn = SimpleDnn(Encoder(input_dim = input_dim, 
                                       output_dim = output_dim), 
                                task_type = 'classification')
    
    csv_logger = CSVLogger(save_dir=save_dir_logger, name = name_logger)

    logger.info(f'Saving logs to {save_dir_logger}/{name_logger}')
    
    logger.info('Start model training')
    trainer = pl.Trainer(logger=csv_logger, max_epochs = 50, log_every_n_steps=20)
    trainer.fit(model=simple_dnn, train_dataloaders = train_loader, val_dataloaders = val_loader)
    logger.info('Training finished successfully')

def main() -> None:
    parser = argparse.ArgumentParser(
                    prog='Train models',
                    description='Train models with synthetic data and different bugs')
    
    parser.add_argument('--save_dir_logger', 
                        type = str, 
                        required = True)
    parser.add_argument('--name_logger', 
                        type = str, 
                        required = True)    
    parser.add_argument('--output_file_name', 
                        type = str, 
                        required = True) 
    parser.add_argument('--data_type', 
                        type = str, 
                        choices=['linear', 'nonlinear'],
                        required = True) 
    parser.add_argument('--break_activations', 
                        type = bool, 
                        default = False) 
    parser.add_argument('--freeze_weights', 
                        type = bool, 
                        default = False)
    parser.add_argument('--freeze_bias', 
                        type = bool, 
                        default = False)
    parser.add_argument('--break_dataloader',
                        type = bool, 
                        default = False)
    args = parser.parse_args()  



    if args.data_type == 'linear':
        train_loader, val_loader = generate_linear_dataloaders()
    elif args.data_type == 'nonlinear':
        train_loader, val_loader  = generate_nonlinear_dataloaders(args.break_dataloader)
        output_dim = 2
    
    torch.manual_seed(4573)

    train_model(train_loader = train_loader,
                val_loader = val_loader,
                save_dir_logger = args.save_dir_logger,
                name_logger = args.name_logger,
                break_activations = args.break_activations,
                output_dim = output_dim,
                freeze_weights= args.freeze_weights,
                freeze_bias = args.freeze_bias
                )

    logger.info('Reformat metrics...')

    input_path = Path(args.save_dir_logger) / args.name_logger / 'version_0' / 'metrics.csv'
    output_path = Path('reformatted_metrics') / (args.output_file_name + '.csv')
    format_csv(input_path, ['train_loss', 'val_loss'], output_path)

    logger.info(f'... finished and written to reformatted_metrics/{output_path}.csv')


if __name__ == "__main__":
    main()

    