import torch
from jaxtyping import Float

from .utils import construct_fnn

"""
batch_size x L x 1536
- transform ->
batch_size x 1536 x L x 1
"""


class MinimalCNN(torch.nn.Module):
    """
    1D convolution followed by two dense layers, akin to biotrainer's offering
    Attributes:
        input_dim: Size of the input vectors (e.g. 1024 for ProtT5 embeddings, 2560 for ESM-2 embeddings). Default: 1280
        output_dim: Size of the output vector (e.g. 20 for GEMME scores). Default: 20
        n_channels: Number of channels. Default: 256
        kernel_size: Size of the convolving kernel, Default: 7
        padding: Amount of padding applied to the input. Default: 3
        fnn_hidden_layers: Dimensions of two dense hidden layers. Default: [256, 64]
        activation_function: Activation function to use for the hidden layers. Default: LeakyReLU
        output_activation_function: Activation function to use for the output layer, e.g. None for linear regression,
            Sigmoid for logistic regression. Default: None
        cnn_dropout_rate: Dropout rate to apply after every layer, if desired. Default: None
        fnn_dropout_rate: Dropout rate to apply after every layer, if desired. Default: None
    Examples:
        gemme_esm2_cnn = MinimalCNN()
    """

    def __init__(
        self,
        input_dim: int = 2560,
        output_dim: int = 20,
        n_channels: int = 256,
        kernel_size=7,
        padding=3,
        fnn_hidden_layers: list[int] = [256, 64],
        activation_function: torch.nn.Module = torch.nn.LeakyReLU,
        output_activation_function: torch.nn.Module = None,
        cnn_dropout_rate: float = None,
        fnn_dropout_rate: float = None,
    ):
        super(MinimalCNN, self).__init__()
        conv_layers = [
            torch.nn.Conv1d(
                input_dim, n_channels, kernel_size=kernel_size, padding=padding
            ),
            activation_function(),
        ]

        if cnn_dropout_rate:
            conv_layers.append(torch.nn.Dropout(cnn_dropout_rate))

        self.conv = torch.nn.Sequential(*conv_layers)

        self.fnn = construct_fnn(
            fnn_hidden_layers,
            n_channels,
            output_dim,
            activation_function,
            output_activation_function,
            fnn_dropout_rate,
        )

    def forward(
        self, X: Float[torch.Tensor, "batch_size length input_dim"]
    ) -> Float[torch.Tensor, "batch_size length output_dim"]:
        X = X.movedim(-1, -2)
        X = self.conv(X)
        X = X.movedim(-1, -2)
        X = self.fnn(X)
        return X.squeeze(-1)


class CombinedCNN(torch.nn.Module):
    # TODO parametrize (CNN parameters, FNN parameters, shared FNN parameters)
    """
    Parallel FNN and CNN whose outputs are concatenated and again fed through dense layers
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 20,
        n_channels: int = 256,
        kernel_size=7,
        padding=3,
        cnn_hidden_layers: list[int] = [64],
        fnn_hidden_layers: list[int] = [256, 64],
        shared_hidden_layers: list[int] = [64],
        activation_function: torch.nn.Module = torch.nn.LeakyReLU,
        output_activation_function: torch.nn.Module = None,
        shared_dropout_rate: float = None,
        cnn_dropout_rate: float = None,
        fnn_dropout_rate: float = None,
    ):
        super(CombinedCNN, self).__init__()
        self.conv = MinimalCNN(
            input_dim=input_dim,
            output_dim=cnn_hidden_layers[-1],
            n_channels=n_channels,
            kernel_size=kernel_size,
            padding=padding,
            fnn_hidden_layers=cnn_hidden_layers[:-1],
            activation_function=activation_function,
            output_activation_function=activation_function,
            cnn_dropout_rate=cnn_dropout_rate,
            fnn_dropout_rate=fnn_dropout_rate,
        )
        self.fnn = construct_fnn(
            hidden_layer_sizes=fnn_hidden_layers[:-1],
            input_dim=input_dim,
            output_dim=fnn_hidden_layers[-1],
            activation_function=activation_function,
            output_activation_function=activation_function,
            dropout_rate=fnn_dropout_rate,
        )
        self.combined = construct_fnn(
            hidden_layer_sizes=shared_hidden_layers,
            input_dim=cnn_hidden_layers[-1] + fnn_hidden_layers[-1],
            output_dim=output_dim,
            activation_function=activation_function,
            output_activation_function=output_activation_function,
            dropout_rate=shared_dropout_rate,
        )

    def forward(self, X):
        X_combined = torch.cat([self.conv(X), self.fnn(X)], dim=-1)
        pred = self.combined(X_combined)
        return pred.squeeze(-1)
