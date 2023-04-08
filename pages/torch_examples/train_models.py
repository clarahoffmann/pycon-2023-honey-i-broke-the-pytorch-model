from loguru import logger
from train_utils import (generate_linear_dataloaders, generate_nonlinear_dataloaders, train_model)
import torch
import argparse
from pathlib import Path
from csv_utils import format_csv

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
    parser.add_argument('--weight_watcher',
                        type = bool, 
                        default = False)
    args = parser.parse_args()  



    if args.data_type == 'linear':
        train_loader, val_loader = generate_linear_dataloaders()
        output_dim = 3
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
                freeze_bias = args.freeze_bias,
                weight_watcher = args.weight_watcher,
                )

    logger.info('Reformat metrics...')

    input_path = Path(args.save_dir_logger) / args.name_logger / 'version_0' / 'metrics.csv'
    output_path = Path('reformatted_metrics') / (args.output_file_name + '.csv')
    format_csv(input_path, ['train_loss', 'val_loss'], output_path)

    logger.info(f'... finished and written to reformatted_metrics/{output_path}')


if __name__ == "__main__":
    main()

    