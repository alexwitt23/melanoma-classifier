#!/usr/bin/env python3
""" Script to export model to onnx format. """

import argparse
import pathlib

import torch

from src import model, dataset


def convert_model(input_model: model.SkinModel, output_dir: pathlib.Path) -> None:

    # Fake input for operater trace
    x = torch.randn(1, 3, 450, 450, requires_grad=True)

    # Export the model
    torch.onnx.export(
        input_model,
        x,
        output_dir / "test.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to ONNX format."
    )
    parser.add_argument(
        "--input_model_path",
        required=True,
        type=pathlib.Path,
        help="Path to saved PyTorch model.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        type=pathlib.Path,
        help="Where to save the exported ONNX model.",
    )
    args = parser.parse_args()

    input_model = model.SkinModel(len(dataset._DATA_CLASSES))
    input_model.load_state_dict(
        torch.load(args.input_model_path.expanduser(), map_location="cpu")
    )
    input_model.eval()

    convert_model(input_model, args.save_dir.expanduser())
