"""Run weightwatcher on trained PyTorch model"""
import argparse
from pathlib import Path

import torch
import weightwatcher as ww
from loguru import logger


def run_weightwatcher(ckpt_path: Path, output_file_path: Path) -> None:
    """Run weightwatcher on PyTorch checkpoint"""

    logger.info(f"Reading checkpoint {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    print(ckpt["state_dict"])
    logger.info("... successfully loaded")

    logger.info("Start weightwatcher analysis ...")
    watcher = ww.WeightWatcher(model=ckpt["state_dict"])
    details = watcher.get_details()
    logger.info("... successfully analyzed")

    logger.info(f"Writing output to {output_file_path} ...")
    details.to_csv(output_file_path, index=False)
    logger.info("... finished writing output")


def main():
    """Run weightwatcher on PyTorch checkpoint"""
    parser = argparse.ArgumentParser(
        prog="Run weightwatcher",
        description="Run weightwatcher on PyTorch checkpoint",
    )

    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_file_path", type=Path, required=True)

    args = parser.parse_args()

    run_weightwatcher(args.ckpt_path, args.output_file_path)


if __name__ == "__main__":
    main()
