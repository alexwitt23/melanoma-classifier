#!/usr/bin/env python3
""" Script to train model to classify melanoma. """

import argparse
import pathlib
import random

import torch
import numpy as np

from src import dataset
from third_party.efficientdet import efficientnet

_SAVE_DIR = pathlib.Path("~/runs/melanoma-model").expanduser()
_LOG_INTERVAL = 5


def train(
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
) -> None:
    """ Main trainer loop. """

    highest_acc = 0
    losses = []
    for epoch in range(40):

        for idx, (img, label) in enumerate(train_loader):

            optimizer.zero_grad()

            if torch.cuda.is_available():
                img = img.float().cuda()
                label = label.cuda()

            # Perform forward pass.
            out = model(img)
            # Compute the loss.
            loss = loss_fn(out, label)
            losses.append(loss.item())

            # Send the loss backwards and compute the gradients in the model.
            loss.backward()
            # Update the model params.
            optimizer.step()
            # Update the learning rate.
            lr_scheduler.step()

            if idx % _LOG_INTERVAL == 0:
                print(f"Epoch {epoch}. Step {idx}. Loss: {np.mean(losses):.5}")

        num_right, total = 0, 0
        model.eval()
        for img, label in eval_loader:

            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)

            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            num_right += (predicted == label).sum().item()
        model.train()

        print(f"Epoch {epoch}, Accuracy: {num_right / total:.2}")

        if highest_acc < num_right / total:
            highest_acc = num_right / total

            torch.save(
                skin_model.state_dict(), _SAVE_DIR / f"model-{highest_acc:.3}.pt"
            )
            print(
                f"Saving model with new highest accuracy, {highest_acc:.3} to {_SAVE_DIR}."
            )


if __name__ == "__main__":
    torch.random.manual_seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description="Melanoma model training code.")
    parser.add_argument(
        "--dataset_dir",
        type=pathlib.Path,
        default=pathlib.Path("~/datasets/melanoma"),
        help="Path to the directory containing both the `train` and `eval` folders.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="efficientnet-b0",
        help="Path to the directory containing both the `train` and `eval` folders.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Train and eval batch size.",
    )
    args = parser.parse_args()

    # Make sure the dataset dir exists.
    data_dir = args.dataset_dir.expanduser()
    assert data_dir.is_dir(), f"Cant not find {data_dir}."

    _SAVE_DIR.mkdir(exist_ok=True, parents=True)
    model_params = efficientnet._MODEL_SCALES["efficientnet-b0"]

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(data_dir / "train", img_size=model_params[2]),
        batch_size=args.batch_size,
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(data_dir / "eval", img_size=model_params[2]),
        batch_size=args.batch_size,
        pin_memory=True,
    )

    model = efficientnet.EfficientNet(
        "efficientnet-b0", num_classes=len(dataset._DATA_CLASSES), img_size=(224, 224)
    )
    if torch.cuda.is_available():
        model.cuda()

    # Create the optimzier
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)

    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader.dataset), eta_min=1e-9
    )

    train(train_loader, eval_loader, model, optimizer, lr_scheduler, loss_fn)
