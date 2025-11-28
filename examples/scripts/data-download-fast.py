#!/usr/bin/env python3
"""
Optimized script to download and process CelebA dataset faster.
Uses larger batch sizes and more efficient processing for CPU.
"""

import sys
from torchvision.datasets import CelebA
import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision.transforms import PILToTensor
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser(
        description="python script to download datasets which are available with torchvision"
    )

    parser.add_argument(
        "-j", "--nthreads", type=int, default=2, help="number of threads to use"
    )
    parser.add_argument(
        "-b", "--batchsize", type=int, default=64, help="batch_size for loading"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("."),
        help="the base folder in which to store the output",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing). If None, processes all."
    )
    parser.add_argument(
        "dataset",
        nargs="+",
        help="datasets to download (possible values: MNIST, CelebA, CIFAR10)",
    )
    args = parser.parse_args()
    
    if not "dataset" in args:
        print("dataset argument not found in", args)
        parser.print_help()
        return 1

    tv_datasets = {"mnist": None, "celeba": CelebA, "cifar10": None}
    rootdir = args.outdir
    if not rootdir.exists():
        print(f"creating root folder {rootdir}")
        rootdir.mkdir(parents=True)

    for dname in args.dataset:
        if dname.lower() not in tv_datasets.keys() or tv_datasets[dname.lower()] is None:
            print(f"{dname} not available for download yet. skipping.")
            continue

        dfolder = rootdir / dname
        dataset = tv_datasets[dname]
        
        if "celeba" in dname.lower():
            train_kwarg = {"split": "train"}
            val_kwarg = {"split": "valid"}
        else:
            train_kwarg = {"train": True}
            val_kwarg = {"train": False}

        print(f"\nProcessing {dname.upper()} dataset...")
        print(f"Downloading and processing train split...")
        
        train_data = dataset(
            dfolder, download=True, transform=PILToTensor(), **train_kwarg
        )
        
        # Limit dataset size if specified
        if args.max_samples and len(train_data) > args.max_samples:
            print(f"Limiting to {args.max_samples} samples for faster processing...")
            indices = torch.randperm(len(train_data))[:args.max_samples]
            train_data = torch.utils.data.Subset(train_data, indices)
        
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=args.batchsize, 
            shuffle=False, 
            num_workers=args.nthreads,
            pin_memory=False  # Disable pin_memory for CPU
        )

        train_batches = []
        for b, (x, y) in enumerate(tqdm(train_loader, desc="Processing train")):
            train_batches.append(x.clone().detach().numpy())

        print(f"Downloading and processing validation split...")
        val_data = dataset(dfolder, download=True, transform=PILToTensor(), **val_kwarg)
        
        # Limit dataset size if specified
        if args.max_samples and len(val_data) > args.max_samples // 10:
            max_val = args.max_samples // 10
            print(f"Limiting validation to {max_val} samples...")
            indices = torch.randperm(len(val_data))[:max_val]
            val_data = torch.utils.data.Subset(val_data, indices)
        
        val_loader = torch.utils.data.DataLoader(
            val_data, 
            batch_size=args.batchsize, 
            shuffle=False, 
            num_workers=args.nthreads,
            pin_memory=False
        )
        val_batches = []
        for b, (x, y) in enumerate(tqdm(val_loader, desc="Processing val")):
            val_batches.append(x.clone().detach().numpy())

        train_x = np.concatenate(train_batches)
        np.savez_compressed(dfolder / "train_data.npz", data=train_x)
        print(
            "Wrote ",
            dfolder / "train_data.npz",
            f"(shape {train_x.shape}, {train_x.dtype})",
        )
        val_x = np.concatenate(val_batches)
        np.savez_compressed(dfolder / "eval_data.npz", data=val_x)
        print(
            "Wrote ", dfolder / "eval_data.npz", f"(shape {val_x.shape}, {val_x.dtype})"
        )

    return 0


if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

