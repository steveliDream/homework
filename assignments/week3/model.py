from typing import Callable
import torch
import torch.nn as nn


class MLP(nn.Module):
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

        # Initialize layers of MLP
        self.layers = nn.ModuleList()

        # Append input layer
        self.layers += [nn.Linear(input_size, hidden_size)]

        # Loop over layers and create hidden layers
        for i in range(hidden_count - 1):
            self.layers += [nn.Linear(input_size, hidden_size)]

        # Create final layer
        self.out = nn.Linear(hidden_size, num_classes)
        self.activation = activation

        for layer in self.layers:
            initializer(layer)

        initializer(self.out)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.out(x)

        return x
