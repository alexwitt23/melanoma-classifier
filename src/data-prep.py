#!/usr/bin/env python3

""" Code used to prep the dataset into simpler layout
ham10k_path: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
"""

import argparse
import pathlib
import random
import shutil

import pandas as pd
import tqdm


def create_dataset(ham10k_path: pathlib.Path, save_dir: pathlib.Path) -> None:

    # Output dirs
    train_dir = save_dir / "train"
    eval_dir = save_dir / "eval"

    train_dir.mkdir(exist_ok=True, parents=True)
    eval_dir.mkdir(exist_ok=True, parents=True)

    csv = pd.read_csv(ham10k_path / "HAM10000_metadata.csv")
    for row in csv.iterrows():
        img_num = row[1]["image_id"].split("_")[1]
        img = list(ham10k_path.rglob(f"*{row[1]['image_id']}.jpg"))[0]

        # 80 / 20 train/test split
        if random.randint(0, 100) < 20:
            shutil.copy2(img, eval_dir / f"{row[1]['dx']}_{img_num}.jpg")
        else:
            shutil.copy2(img, train_dir / f"{row[1]['dx']}_{img_num}.jpg")


if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser(description="Prepare the data for training.")
    parser.add_argument(
        "--ham10k_path",
        type=pathlib.Path,
        required=False,
        help="Path to the HAM10000 extracted zip.",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        required=False,
        default=pathlib.Path("~/datasets/melanoma"),
        help="Path to save the train/val data to.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir.expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)

    ham10k_path = None
    if args.ham10k_path is not None:
        ham10k_path = args.ham10k_path.expanduser()
        assert ham10k_path.is_dir(), f"Can't find {ham10k_path}."

    create_dataset(ham10k_path, save_dir)
