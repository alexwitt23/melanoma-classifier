""" A custom model playground meant for understanding
the underlying concepts behind ML models. """

import torch 


#TODO(paul)
class CustomModel(torch.nn.Module):
    """ Typically, in PyTorch, models or custom layers will
    inherit from torch.nn.Module so the models/layers can be
    used fluidly with the rest of the python API. """
    def __init__(self) -> None:
        super().__init__()

        # Construct the model here. Use the Sequential module
        # so all the layers will be constructed in a sequential 
        # format. Passing an input in the model, flows through 
        # all the layers.
        self.layers = torch.nn.Sequential()

    # Define the __call__ method so we can call the object
    # like a function. Not necessary, but looks clean.
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass 