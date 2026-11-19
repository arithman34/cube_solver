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
