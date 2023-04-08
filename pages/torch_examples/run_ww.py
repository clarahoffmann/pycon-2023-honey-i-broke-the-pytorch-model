import torch
import weightwatcher as ww
from pathlib import Path
import argparse
from loguru import logger
import pandas as pd


def run_weightwatcher(ckpt_path: Path, output_file_path: Path) -> None:

    logger.info(f'Reading checkpoint {ckpt_path} ...')
    ckpt = torch.load(ckpt_path,map_location=torch.device('cpu') )
    print(ckpt['state_dict'])
    logger.info(f'... successfully loaded')

    logger.info(f'Start weightwatcher analysis ...')
    watcher = ww.WeightWatcher(model=ckpt['state_dict'])
    results = watcher.analyze()
    details = watcher.get_details()
    logger.info(f'... successfully analyzed')

    print(details)

    logger.info(f'Writing output to {output_file_path} ...')
    details.to_csv(output_file_path, index = False)
    logger.info(f'... finished writing output')


def main():
    parser = argparse.ArgumentParser(
                    prog='Train models',
                    description='Train models with synthetic data and different bugs')
    
    parser.add_argument('--ckpt_path', 
                        type = Path, 
                        required = True)
    parser.add_argument('--output_file_path', 
                        type = Path, 
                        required = True)

    args = parser.parse_args()  

    run_weightwatcher(args.ckpt_path, args.output_file_path)

if __name__ == '__main__':
    main()