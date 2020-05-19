#!/usr/bin/env python3
""" Simple script used to load a model and classify an image. """

import argparse
import pathlib

import torch
import cv2

from src import dataset
from third_party.efficientdet import efficientnet


def inference_image(
    model_path: pathlib.Path, model_type: str, image_path: pathlib.Path
) -> None:
    """ Perform inference over a single image. """
    assert model_path.is_file(), f"{model_path} can not be found!"
    assert image_path.is_file(), f"{image_path} can not be found!"

    model = efficientnet.EfficientNet(
        model_type, num_classes=len(dataset._DATA_CLASSES), img_size=(224, 224)
    )
    model.load(torch.load(model_path, map_location="cpu"))

    if torch.cuda.is_available():
        model.cuda()
        model.half()

    img = cv2.imread(str(img_path))
    assert img is not None, f"Trouble reading {img_path}!"

    # Perform augmentations
    img_tensor = torch.from_numpy(dataset.eval_augmentations(image=img)["image"])

    if torch.cuda.is_available:
        img_tensor = img_tensor.cuda()

    results = model(img_tensor)

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to inference an input image and print the result."
    )
    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        required=True,
        help="Path to the saved model file to load.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The efficientnet model type."
    )
    parser.add_argument(
        "--image_path",
        type=pathlib.Path,
        required=True,
        help="Path to the image to inference.",
    )
    args = parser.parse_args()

    inference_image(
        args.model_path.expanduser(), args.model_type, args.image_path.expanduser()
    )
