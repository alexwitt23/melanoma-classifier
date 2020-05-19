#!/usr/bin/env python3
""" Script to train model to classify melanoma. """

import pathlib
import random

import torch
import numpy as np

from src import dataset, model, efficientnet

_DATA_DIR = pathlib.Path("~/datasets/melanoma").expanduser()
_SAVE_DIR = pathlib.Path("~/runs/melanoma-model").expanduser()


def train(
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    skin_model,
    optimizer,
    loss_fn,
) -> None:
    highest_acc = 0
    losses = []
    for epoch in range(40):

        skin_model.train()
        for idx, (img, label) in enumerate(train_loader):

            optimizer.zero_grad()
            if torch.cuda.is_available():
                img = img.float().cuda()
                label = label.cuda()

            out = skin_model(img)
            loss = loss_fn(out, label)
            losses.append(loss.item())
            # Send the loss backwards and compute the gradients in the model
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean(losses):.5}")

        num_right = 0
        total = 0
        skin_model.eval()
        for img, label in eval_loader:

            if torch.cuda.is_available():
                img = img.float().cuda()

            out = skin_model(img).cpu()

            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            num_right += (predicted == label).sum().item()

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

    _SAVE_DIR.mkdir(exist_ok=True, parents=True)
    model_params = efficientnet._MODEL_SCALES["efficientnet-b0"]
    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(_DATA_DIR / "train", img_size=model_params[2]),
        batch_size=64,
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(_DATA_DIR / "eval", img_size=model_params[2]),
        batch_size=64,
        pin_memory=True,
    )
    # Instantiate model
    test_model = efficientnet.EfficientNet(
        "efficientnet-b0", num_classes=len(dataset._DATA_CLASSES), img_size=(224, 224)
    )
    if torch.cuda.is_available():
        test_model.cuda()
    # Create the optimzier
    optimizer = torch.optim.RMSprop(test_model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    train(train_loader, eval_loader, test_model, optimizer, loss_fn)
