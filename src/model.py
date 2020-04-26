""" Model to train. """

import torch
import torchvision


class SkinModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.model = torchvision.models.resnet34(
            pretrained=False, num_classes=num_classes
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
