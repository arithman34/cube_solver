import argparse
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from cube.solver import solve
from cube.state import Cube
from training.data import apply_scrambles, scramble_moves
from training.encoding import encode_stickers
from training.model import ValueModel
from training.train import CHECKPOINT_DIR, load_checkpoint

EVAL_SEED = 1_000_000


@dataclass
class DepthResult:
    depth: int
    mean_prediction: float
    mean_abs_error: float
    kociemba_length: float | None = None


def cubes_at_depth(depth: int, n: int, rng: np.random.Generator) -> NDArray[np.uint8]:
    """n cubes, each scrambled exactly `depth` moves."""
    return apply_scrambles(scramble_moves(np.full(n, depth), depth, rng))


def predict(
    model: ValueModel,
    stickers: NDArray[np.uint8],
    device: torch.device,
    mixed_precision: bool = False,
) -> NDArray[np.float32]:
    """Predicted distances as a NumPy array of shape (n,).

    mixed_precision runs the model in bfloat16 on CUDA, which is faster and
    uses less memory. It has no effect on the CPU.
    """
    model.eval()
    use_amp = mixed_precision and device.type == "cuda"
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp),
    ):
        inputs = encode_stickers(stickers, device)
        return model(inputs).float().cpu().numpy()


def evaluate(
    model: ValueModel,
    max_moves: int,
    n_per_depth: int,
    device: torch.device,
    kociemba_per_depth: int = 0,
    seed: int = EVAL_SEED,
) -> list[DepthResult]:
    """One DepthResult per depth from 1 to max_moves."""
    rng = np.random.default_rng(seed)
    results = []
    for depth in range(1, max_moves + 1):
        stickers = cubes_at_depth(depth, n_per_depth, rng)
        predictions = predict(model, stickers, device)

        mean_prediction = float(np.mean(predictions))
        mean_abs_error = float(np.mean(np.abs(predictions - depth)))

        kociemba_length = None
        if kociemba_per_depth > 0:
            lengths = [len(solve(Cube(row))) for row in stickers[:kociemba_per_depth]]
            kociemba_length = float(np.mean(lengths))

        results.append(
            DepthResult(
                depth=depth,
                mean_prediction=mean_prediction,
                mean_abs_error=mean_abs_error,
                kociemba_length=kociemba_length,
            )
        )

    return results


def print_results(results: list[DepthResult]) -> None:
    """Print a table with one row per depth."""
    print("depth  predicted  abs error  kociemba")
    for result in results:
        kociemba = (
            "-" if result.kociemba_length is None else f"{result.kociemba_length:.1f}"
        )
        print(
            f"{result.depth:5}  {result.mean_prediction:9.2f}  {result.mean_abs_error:9.2f}  {kociemba:>8}"
        )


def main() -> None:
    """Load a checkpoint (value_net.pt by default), evaluate it and print the table."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("checkpoint", nargs="?", default="value_net.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.checkpoint} on {device}")

    model = load_checkpoint(CHECKPOINT_DIR / args.checkpoint, device)
    results = evaluate(
        model=model,
        max_moves=20,
        n_per_depth=1000,
        device=device,
        kociemba_per_depth=20,
    )
    print_results(results)


if __name__ == "__main__":
    main()
