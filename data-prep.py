#!/usr/bin/env python3

""" Steps to setup dataset.
1) download dataset from Kaggle and extract it into ~/datasets
2) perform:
    $:~/datasets/skin-cancer-mnist-ham10000 mkdir imgs
    $:~/datasets/skin-cancer-mnist-ham10000 mv ham10000_images_part_*/*.jpg imgs && \
        mv HAM10000_*/*.jpg imgs
3) run this script
4) delete any remaining folders/files in ~/datasets/skin-cancer-mnist-ham10000
"""


import pathlib
import random

import pandas as pd


if __name__ == "__main__":
    # Read in the csv
    data_dir = pathlib.Path("~/datasets/skin-cancer-mnist-ham10000/").expanduser()
    img_dir = data_dir / "imgs"

    csv = pd.read_csv(data_dir / "HAM10000_metadata.csv")

    save_dir = pathlib.Path("~/datasets/melanoma10k").expanduser()

    # Output dirs
    train_dir = save_dir / "train"
    eval_dir = save_dir / "eval"

    train_dir.mkdir(exist_ok=True, parents=True)
    eval_dir.mkdir(exist_ok=True, parents=True)

    for row in csv.iterrows():
        img_name = row[1]["image_id"]

        img_num = row[1]["image_id"].split("_")[1]

        # 80 / 20 train/test split
        if random.randint(0, 100) < 19:
            (img_dir / f"{row[1]['image_id']}.jpg").rename(
                eval_dir / f"{row[1]['dx']}_{img_num}.jpg"
            )
        else:
            (img_dir / f"{row[1]['image_id']}.jpg").rename(
                train_dir / f"{row[1]['dx']}_{img_num}.jpg"
            )
