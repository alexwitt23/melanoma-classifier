#!/usr/bin/env python3
""" Script to train model to classify melanoma. """

import pathlib 
import random

import torch 

from src import dataset

_DATA_DIR = pathlib.Path("~/datasets/melanoma10k").expanduser()







if __name__ == "__main__":

    torch.random.manual_seed(42)
    random.seed(42)

    train_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(_DATA_DIR / "train", img_width=100, img_height=100)
    )

    eval_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(_DATA_DIR / "eval", img_width=100, img_height=100)
    )

    
