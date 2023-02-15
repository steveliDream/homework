import torch
from typing import Callable
import torch
import torch.nn as nn


class MLP(nn.Module):
    """this is the model class for a MLP network

    Args:
        nn (_class_): inherent methods from the nn.Module class
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.hidden_count = hidden_count
        dropout_prob = 0.3

        # Initialize layers of MLP
        self.layers = nn.ModuleList()

        # Append input layer
        self.layers += [nn.Linear(input_size, hidden_size)]

        # Loop over layers and create hidden layers
        for i in range(hidden_count - 1):
            self.layers += [
                nn.Linear(hidden_size, hidden_size),
                activation(),
                nn.Dropout(dropout_prob),
            ]

        # Create final layer
        self.layers += [nn.Linear(hidden_size, num_classes)]
        self.activation = activation

        for layer in self.layers:
            for param in layer.parameters():
                cur_weights = torch.empty_like(param.data)
                if len(cur_weights.size()) == 1:
                    cur_weights = cur_weights.view(-1, 1)
                cur_weights = initializer(cur_weights)
                param.data = cur_weights.squeeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = layer(x)

        return x
