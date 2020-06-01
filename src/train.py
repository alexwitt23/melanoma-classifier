#!/usr/bin/env python3
""" Script to train model to classify various skin lesions. """

import argparse
import pathlib
import random

import torch
import numpy as np
import yaml

from src import dataset
from efficientdet import efficientnet

_SAVE_DIR = pathlib.Path("~/runs/melanoma-model").expanduser()
_LOG_INTERVAL = 5


def train(
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    num_epochs: int,
) -> None:
    """ Main function that will perform training then call out for evaluation. """

    highest_acc = 0
    losses = []
    for epoch in range(num_epochs):

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
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch}. Step {idx}. Loss: {np.mean(losses):.5}. lr: {lr:.5}"
                )

        num_right, total = 0, 0
        model.eval()
        with torch.no_grad():
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

            torch.save(model.state_dict(), _SAVE_DIR / f"model-{highest_acc:.3}.pt")
            print(
                f"Saving model with new highest accuracy, {highest_acc:.3} to {_SAVE_DIR}."
            )


def create_optimizer(
    optim_config: dict, model: torch.nn.Module
) -> torch.optim.Optimizer:
    """ Factory for optimizer creation. """
    optim_type = optim_config.get("type", None).lower()

    if optim_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-4,
            momentum=optim_config.get("momentum", 0.9),
            nesterov=optim_config.get("nesterov", True),
            weight_decay=float(optim_config.get("weight_decay", 1e-5)),
        )
    else:
        raise ValueError(f"Optimizer {optim_type} not supported.")

    return optimizer


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
        "--config", type=pathlib.Path, required=True, help="Config used for training.",
    )
    args = parser.parse_args()

    # Make sure the dataset dir exists.
    data_dir = args.dataset_dir.expanduser()
    assert data_dir.is_dir(), f"Cant not find {data_dir}."

    _SAVE_DIR.mkdir(exist_ok=True, parents=True)

    config_path = args.config.expanduser()
    assert config_path.is_file(), f"Can not find {config_path}."
    config = yaml.safe_load(config_path.read_text())
    model_type = config.get("model", None)
    model_params = efficientnet._MODEL_SCALES[model_type]

    train_config = config.get("training")
    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(data_dir / "train", img_size=model_params[2]),
        batch_size=train_config.get("batch_size", 15),
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(data_dir / "eval", img_size=model_params[2]),
        batch_size=train_config.get("batch_size", 15),
        pin_memory=True,
    )

    model = efficientnet.EfficientNet(
        model_type, num_classes=len(dataset._DATA_CLASSES)
    )
    if torch.cuda.is_available():
        model.cuda()

    # Create the optimzier
    optimizer = create_optimizer(train_config.get("optimizer", None), model)

    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create a learning rate scheduler
    lr_config = train_config.get("lr_schedule", None)
    max_lr = float(lr_config.get("max_lr", 1e-2))
    start_lr = float(lr_config.get("start_lr", 1e-4))
    end_lr = float(lr_config.get("end_lr", 1e-7))
    warmup_epochs = float(lr_config.get("warmup_epochs", 0.1))
    total_steps = len(train_loader) * config.get("epochs", 20)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        div_factor=max_lr / start_lr,
        final_div_factor=start_lr / end_lr,
        pct_start=warmup_epochs / train_config.get("epochs"),
    )

    train(
        train_loader,
        eval_loader,
        model,
        optimizer,
        lr_scheduler,
        loss_fn,
        train_config.get("epochs", 20),
    )
