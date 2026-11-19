import numpy as np
from numpy.typing import NDArray

from cube.faces import TOTAL_STICKERS
from cube.moves import MOVE_NAMES, MOVES
from cube.state import SOLVED_STICKERS

IDENTITY = len(MOVE_NAMES)
MOVE_PERMS = np.stack(
    [MOVES[name] for name in MOVE_NAMES] + [np.arange(TOTAL_STICKERS)]
)
MOVE_FACES = np.array([name[0] for name in MOVE_NAMES] + [""])


def scramble_moves(
    depths: NDArray[np.int64], max_moves: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Random move indices of shape (n, max_moves), one scramble per row."""
    n = len(depths)
    moves = np.full((n, max_moves), IDENTITY, dtype=np.intp)

    for step in range(max_moves):
        active = depths > step
        column = rng.integers(0, IDENTITY, size=n)

        if step > 0:
            previous_faces = MOVE_FACES[moves[:, step - 1]]
            clash = active & (MOVE_FACES[column] == previous_faces)
            while clash.any():
                column[clash] = rng.integers(0, IDENTITY, size=clash.sum())
                clash = active & (MOVE_FACES[column] == previous_faces)

        moves[:, step] = np.where(active, column, IDENTITY)

    return moves


def apply_scrambles(moves: NDArray[np.intp]) -> NDArray[np.uint8]:
    """Apply each row of move indices to a solved cube. Returns (n, 54) stickers."""
    stickers = np.tile(SOLVED_STICKERS, (len(moves), 1))

    for step in range(moves.shape[1]):
        stickers = np.take_along_axis(stickers, MOVE_PERMS[moves[:, step]], axis=1)

    return stickers


def generate(
    n: int, max_moves: int, seed: int | None = None
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Make n scrambled cubes and their scramble lengths (repeatable with a seed)."""
    if max_moves < 1:
        raise ValueError(f"Max moves must be > 0. Got {max_moves}.")

    rng = np.random.default_rng(seed)
    depths = rng.integers(1, max_moves + 1, size=n)
    moves = scramble_moves(depths, max_moves, rng)

    return apply_scrambles(moves), depths
