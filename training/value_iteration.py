import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from cube.moves import MOVE_NAMES
from cube.state import SOLVED_STICKERS
from training.data import generate
from training.encoding import encode_stickers
from training.evaluate import predict
from training.model import ValueModel, build_model
from training.search import expand
from training.train import CHECKPOINT_DIR, save_checkpoint


@dataclass
class ValueIterationConfig:
    iterations: int = 1_000
    cubes_per_iteration: int = 500_000
    batch_size: int = 5_000
    max_moves: int = 30
    model: str = "residual"  # "residual" (DeepCubeA) or "mlp" (ValueNet)
    first_size: int = 5000  # residual only
    hidden_size: int = 1000
    num_blocks: int = 4  # residual only
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4  # decays exponentially to this by the end
    mixed_precision: bool = True
    update_threshold: float = 0.05  # update target_model once the loss is below this
    update_every: int = 4  # or after this many iterations without an update
    save_every: int = 10
    seed: int = 0


def bellman_targets(
    target_model: ValueModel,
    stickers: NDArray[np.uint8],
    device: torch.device,
    chunk_size: int = 2_000,
    mixed_precision: bool = False,
) -> NDArray[np.float32]:
    """1 + the smallest predicted distance among each cube's 18 neighbours.

    Solved neighbours count as 0, and solved cubes get a target of 0.
    """
    all_targets = []
    for start in range(0, len(stickers), chunk_size):
        chunk = stickers[start : start + chunk_size]
        children = expand(chunk)
        values = predict(target_model, children, device, mixed_precision)
        values[(children == SOLVED_STICKERS).all(axis=1)] = 0
        targets = 1 + values.reshape(len(chunk), len(MOVE_NAMES)).min(axis=1)
        targets[(chunk == SOLVED_STICKERS).all(axis=1)] = 0
        all_targets.append(targets)

    return np.concatenate(all_targets)


def value_iteration_step(
    model: ValueModel,
    target_model: ValueModel,
    optimizer: torch.optim.Optimizer,
    config: ValueIterationConfig,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    """Train on one set of fresh cubes against Bellman targets.

    Returns the mean loss and the mean target.
    """
    stickers, _ = generate(config.cubes_per_iteration, config.max_moves, seed)
    targets = bellman_targets(
        target_model, stickers, device, mixed_precision=config.mixed_precision
    )

    inputs = encode_stickers(stickers, device)
    labels = torch.from_numpy(targets).to(device)

    use_amp = config.mixed_precision and device.type == "cuda"
    model.train()
    total_loss = 0.0
    for start in range(0, len(inputs), config.batch_size):
        batch_inputs = inputs[start : start + config.batch_size]
        batch_labels = labels[start : start + config.batch_size]

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            loss = F.mse_loss(model(batch_inputs), batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_inputs)

    return total_loss / len(inputs), float(targets.mean())


def should_update_target(
    loss: float, since_update: int, config: ValueIterationConfig
) -> bool:
    """Update once the model fits the current targets, or after too long without."""
    return loss < config.update_threshold or since_update >= config.update_every


def decay_rate(config: ValueIterationConfig) -> float:
    """Per-iteration factor that takes learning_rate to final_learning_rate.

    The first iteration uses learning_rate and the last uses final_learning_rate.
    """
    if config.iterations <= 1:
        return 1.0
    ratio = config.final_learning_rate / config.learning_rate
    return ratio ** (1 / (config.iterations - 1))


def value_iteration(
    config: ValueIterationConfig, device: torch.device, path: Path | None = None
) -> ValueModel:
    """Train a value network from nothing but the solved state.

    With a path, a checkpoint is saved there every save_every target updates,
    so a long run can be stopped at any point without losing everything.
    """
    torch.manual_seed(config.seed)
    model = build_model(asdict(config)).to(device)
    target_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay_rate(config))

    updates = 0
    since_update = 0
    for iteration in range(config.iterations):
        loss, mean_target = value_iteration_step(
            model, target_model, optimizer, config, config.seed + iteration, device
        )
        since_update += 1

        message = (
            f"Iteration {iteration + 1}/{config.iterations}: "
            f"loss {loss:.4f}, mean target {mean_target:.2f}, "
            f"lr {scheduler.get_last_lr()[0]:.1e}"
        )
        scheduler.step()
        if should_update_target(loss, since_update, config):
            target_model.load_state_dict(model.state_dict())
            updates += 1
            since_update = 0
            message += f", target update {updates}"

            if path is not None and updates % config.save_every == 0:
                save_checkpoint(model, config, path)
                message += ", saved"
        print(message)

    return model


def main() -> None:
    """Train and save to training/checkpoints (value_iteration.pt by default)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("checkpoint", nargs="?", default="value_iteration.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    config = ValueIterationConfig()
    path = CHECKPOINT_DIR / args.checkpoint
    model = value_iteration(config, device, path)

    save_checkpoint(model, config, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
