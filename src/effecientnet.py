""" Code to implement an effecientnet in PyTorch. 

The architecture is based on scaling four model parameters: 
depth (more layers), width (more filters per layer), resolution 
(larger input images), and dropout (a regularization technique 
to cause sparse feature learning). """

from typing import Tuple
import math

import torch

_MODEL_SCALES = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5),
}

# These are the default parameters for the model's mobile inverted residual
# bottleneck layers
_DEFAULT_MB_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]


class Swish(torch.nn.Module):
    """ Swish activation function presented here:
    https://arxiv.org/pdf/1710.05941.pdf. """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return x * torch.sigmoid(x)


# TODO(alex) this is confusing. Copied right from original implemenation.
# This has to do with keeping the scaling consistent, but I don't understand
# exactly whats going on.
def round_filters(
    filters: int, scale_params: Tuple[float, float, int, float], min_depth: int = 8
) -> int:
    """ Determine the number of filters based on the depth multiplier. """

    filters *= scale_params[0]
    new_filters = max(min_depth, int(filters + min_depth / 2) // min_depth * min_depth)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += min_depth

    return int(filters)

# TODO(alex) this is also confusing
def round_repeats(repeats: int, depth_multiplier: int):
    if depth_multiplier == 1.0:
        return repeats
    return int(math.ceil(depth_multiplier * repeats))


class PointwiseConv(torch.nn.Module):
    """ A pointwise convolutional layer. This apply a 1 x 1 x N filter
    to the input to quickly expand the input without many parameters. """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            Swish(),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DepthwiseConv(torch.nn.Module):
    """ A depthwise convolutions with has only one filter per incoming channel
    number. The output channels number is the same and the input. """

    def __init__(self, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        # Calculate the padding to keep the input 'same' dimensions. Add
        # padding to the right and bottom first.
        diff = kernel_size - stride
        padding = [
            diff // 2,
            diff - diff // 2,
            diff // 2,
            diff - diff // 2,
        ]
        self.layers = torch.nn.Sequential(
            torch.nn.ZeroPad2d(padding),
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=channels,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=channels),
            Swish(),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SqueezeExcitation(torch.nn.Module):
    """  See here for one of the original implementations:
    https://arxiv.org/pdf/1709.01507.pdf. The layer 'adaptively recalibrates 
    channel-wise feature responses by explicitly  modeling interdependencies
    between channels.' """

    def __init__(self, in_channels: int, out_channels: int, se_ratio: float) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),  # 1 x 1 x in_channels
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(out_channels * se_ratio),
                kernel_size=1,
            ),
            Swish(),
            torch.nn.Conv2d(
                in_channels=int(out_channels * se_ratio),
                out_channels=out_channels,
                kernel_size=1,
            ),
            torch.nn.Sigmoid(),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply the squeezing and excitation, then elementwise multiplacation of 
        the excitation 1 x 1 x out_channels tensor. """
        return x * self.layers(x)


class MBConvBlock(torch.nn.Module):
    """ Mobile inverted residual bottleneck layer. """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expand_ratio: int,
        stride: int,
        se_ratio: int,
    ) -> None:
        super().__init__()
        diff = kernel_size - stride
        padding = [
            diff // 2,
            diff - diff // 2,
            diff // 2,
            diff - diff // 2,
        ]
        self.layers = torch.nn.Sequential(
            PointwiseConv(in_channels=in_channels, out_channels=out_channels),
            DepthwiseConv(
                channels=out_channels, kernel_size=kernel_size, stride=stride
            ),
            SqueezeExcitation(
                in_channels=out_channels, out_channels=out_channels, se_ratio=se_ratio,
            ),
            torch.nn.ZeroPad2d(padding),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EffecientNet(torch.nn.Module):
    """ Entrypoint for creating an effecientnet. """

    def __init__(
        self, scale_params: Tuple[float, float, int, float], num_classes: int
    ) -> None:
        """ Instantiant the EffecientNet. 

        Args:
            scale_params: (width_coefficient, depth_coefficient, resolution, dropout_rate)
        """

        super().__init__()

        # Add the first layer, a simple 3x3 filter conv layer.
        out_channels = round_filters(32, scale_params)
        self.model = torch.nn.ModuleList(
            [
                torch.nn.ZeroPad2d([0, 1, 0, 1]),
                torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(out_channels),
                Swish(),
            ]
        )
        in_channels = out_channels
        # Now loop over the MBConv layer params
        for mb_params in _DEFAULT_MB_BLOCKS_ARGS:
            out_channels = round_filters(
                filters=mb_params["filters_out"], scale_params=scale_params
            )
            repeats = round_repeats(mb_params["repeats"], scale_params[1])
            for _ in range(repeats):
                self.model += [
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=mb_params["kernel_size"],
                        expand_ratio=mb_params["expand_ratio"],
                        stride=mb_params["strides"],
                        se_ratio=mb_params["se_ratio"],
                    )
                ]
                in_channels = out_channels

        # Construct the last layer
        out_channels = round_filters(1280, scale_params=scale_params)
        self.model += [
            PointwiseConv(in_channels=in_channels, out_channels=out_channels),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Dropout(p=scale_params[-1]),
        ]

        self.model_layers = torch.nn.Sequential(*self.model)
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=out_channels, out_features=num_classes),
        )

        for module in self.model_layers.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        features = self.model_layers(x)
        features = features.view(features.shape[0], -1)
        return self.classification_head(features)
