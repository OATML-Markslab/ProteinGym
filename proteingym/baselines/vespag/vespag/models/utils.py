from copy import deepcopy

import torch


def construct_fnn(
    hidden_layer_sizes: list[int],
    input_dim: int = 1024,
    output_dim: int = 20,
    activation_function: torch.nn.Module = torch.nn.LeakyReLU,
    output_activation_function: torch.nn.Module = None,
    dropout_rate: float = None,
):
    layer_sizes = deepcopy(hidden_layer_sizes)

    layer_sizes.insert(0, input_dim)
    layer_sizes.append(output_dim)
    layers = []
    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        layers.append(torch.nn.Linear(in_size, out_size))
        if dropout_rate:
            layers.append(torch.nn.Dropout(dropout_rate))
        layers.append(activation_function())

    # remove last activation function
    layers = layers[:-1]

    # remove last dropout if applicable
    if dropout_rate:
        layers = layers[:-1]

    if output_activation_function:
        layers.append(output_activation_function())

    return torch.nn.Sequential(*layers)


class MeanModel(torch.nn.Module):
    def __init__(self, *models: torch.nn.Module):
        super(MeanModel, self).__init__()
        self.models = list(models)

    def forward(self, x):
        return sum([model(x) for model in self.models]) / len(self.models)
