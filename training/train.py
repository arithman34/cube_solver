from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from training.data import generate
from training.encoding import encode_stickers
from training.model import ValueModel, ValueNet, build_model

if TYPE_CHECKING:
    from training.value_iteration import ValueIterationConfig

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


@dataclass
class TrainConfig:
    epochs: int = 50
    cubes_per_epoch: int = 100_000
    batch_size: int = 1_000
    max_moves: int = 20
    hidden_size: int = 512
    learning_rate: float = 1e-3
    seed: int = 0


def train_epoch(
    model: ValueNet,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    seed: int,
    device: torch.device,
) -> float:
    """Train on one set of freshly generated cubes and return the mean loss."""
    stickers, depths = generate(config.cubes_per_epoch, config.max_moves, seed)

    inputs = encode_stickers(stickers, device)
    labels = torch.from_numpy(depths).float().to(device)

    model.train()
    total_loss = 0.0
    for start in range(0, len(inputs), config.batch_size):
        batch_inputs = inputs[start : start + config.batch_size]
        batch_labels = labels[start : start + config.batch_size]

        loss = F.mse_loss(model(batch_inputs), batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_inputs)

    return total_loss / len(inputs)


def train(config: TrainConfig, device: torch.device) -> ValueNet:
    """Train a new ValueNet from scratch with the given settings."""
    torch.manual_seed(config.seed)
    model = ValueNet(hidden_size=config.hidden_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        loss = train_epoch(model, optimizer, config, config.seed + epoch, device)
        print(f"Epoch {epoch + 1}/{config.epochs}: loss {loss:.3f}")

    return model


def save_checkpoint(
    model: ValueModel, config: "TrainConfig | ValueIterationConfig", path: Path
) -> None:
    """Save the weights together with the settings that produced them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path, device: torch.device) -> ValueModel:
    """Rebuild the network saved by save_checkpoint, whichever kind it is."""
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    config = TrainConfig()
    model = train(config, device)

    path = CHECKPOINT_DIR / "value_net.pt"
    save_checkpoint(model, config, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
