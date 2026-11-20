from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from training.encoding import ENCODED_SIZE


class ValueNet(nn.Module):
    """MLP that predicts how many moves a cube is from solved."""

    def __init__(self, hidden_size: int = 512) -> None:
        """Two hidden layers of hidden_size units with ReLU."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ENCODED_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """(n, 324) encoded cubes -> (n,) predicted distances."""
        return self.layers(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """Two linear layers whose output is added back onto the block's input."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.layers(x))


class ResidualValueNet(nn.Module):
    """DeepCubeA-style network: a wide first layer, then residual blocks.

    The defaults match DeepCubeA (5000, 1000, 4 blocks, about 14.7M parameters).
    """

    def __init__(
        self, first_size: int = 5000, hidden_size: int = 1000, num_blocks: int = 4
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(ENCODED_SIZE, first_size),
            nn.BatchNorm1d(first_size),
            nn.ReLU(),
            nn.Linear(first_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """(n, 324) encoded cubes -> (n,) predicted distances, always in float32.

        Under mixed precision the last layer still runs in float32, so distances
        like 15.3 are not rounded to the nearest 1/16.
        """
        x = self.blocks(self.input(x))
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.output(x.float()).squeeze(-1)


ValueModel = ValueNet | ResidualValueNet


def build_model(config: dict[str, Any]) -> ValueModel:
    """Build the network a saved config describes.

    Configs without a "model" key come from supervised training and use ValueNet.
    """
    if config.get("model", "mlp") == "residual":
        return ResidualValueNet(
            first_size=config["first_size"],
            hidden_size=config["hidden_size"],
            num_blocks=config["num_blocks"],
        )
    return ValueNet(hidden_size=config["hidden_size"])
