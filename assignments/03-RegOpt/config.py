from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 128
    num_epochs = 10
    initial_learning_rate = 2e-3
    initial_weight_decay = 0

    lrs_kwargs = {
        "T_max": 10,
        "eta_min": 0,
        "last_epoch": -1,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
            Normalize((0, 0, 0), (1, 1, 1)),
        ]
    )
