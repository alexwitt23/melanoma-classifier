#!/usr/bin/env python3
""" Script to train model to classify melanoma. """

import pathlib
import random

import torch
import numpy as np

from src import dataset, model

_DATA_DIR = pathlib.Path("~/datasets/melanoma10k").expanduser()
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
    for epoch in range(20):

        for img, label in train_loader:

            skin_model.train()
            optimizer.zero_grad()

            img = img.float().cuda()
            label = label.cuda()

            out = skin_model(img)
            loss = loss_fn(out, label.cuda())
            losses.append(loss.item())
            # Send the loss backwards and compute the gradients in the model
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {np.mean(losses):.5}")

        num_right = 0
        total = 0
        for img, label in eval_loader:

            skin_model.eval()
            img = img.float().cuda()
            out = skin_model(img).cpu()

            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            num_right += (predicted == label).sum().item()

        print(f"Epoch {epoch}, Accuracy: {num_right / total:.2}")

        if highest_acc < num_right / total:
            highest_acc = num_right / total

            torch.save(skin_model.state_dict(), _SAVE_DIR / f"model-{highest_acc:.3}.pt")
            print(
                f"Saving model with new highest accuracy, {highest_acc:.3} to {_SAVE_DIR}."
            )


if __name__ == "__main__":

    torch.random.manual_seed(42)
    random.seed(42)

    _SAVE_DIR.mkdir(exist_ok=True, parents=True)

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(_DATA_DIR / "train"), batch_size=32, pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(_DATA_DIR / "eval"), batch_size=32, pin_memory=True,
    )
    # Instantiate model
    test_model = model.SkinModel(len(dataset._DATA_CLASSES))
    # Create the optimzier
    optimizer = torch.optim.SGD(
        test_model.parameters(), lr=1e-2, weight_decay=1e-4, momentum=0.9, nesterov=True
    )
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    test_model.cuda()
    train(train_loader, eval_loader, test_model, optimizer, loss_fn)
