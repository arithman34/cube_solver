import numpy as np

from cube.moves import MOVE_NAMES
from cube.state import Cube


def generate(
    n: int, max_moves: int, seed: int | None = None
) -> tuple[list[str], list[int]]:
    """Make n scrambled cubes and their scramble lengths (repeatable with a seed)."""
    if max_moves < 1:
        raise ValueError(f"Max moves must be > 0. Got {max_moves}")

    rng = np.random.default_rng(seed)

    cubes = []
    depths = []
    for _ in range(n):
        k = int(rng.integers(1, max_moves + 1))
        moves = scramble_moves(k, rng)
        cubes.append(Cube.solved().apply_moves(moves).to_string())
        depths.append(k)

    return cubes, depths


def scramble_moves(k: int, rng: np.random.Generator) -> list[str]:
    """Return k random moves with no two in a row on the same face."""
    out = []

    while len(out) < k:
        move = str(rng.choice(MOVE_NAMES))

        if out and move[0] == out[-1][0]:
            continue

        out.append(move)

    return out
