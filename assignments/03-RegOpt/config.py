from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 64
    num_epochs = 10
    initial_learning_rate = 0.002
    initial_weight_decay = 0

    lrs_kwargs = {
        "T_max": 10 * 782,
        "eta_min": 2e-9,
        "last_epoch": -1,
        "verbose": False,
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
            Normalize((0.8, 0.6, 0.5), (0.5, 0.3, 0.2)),
        ]
    )
