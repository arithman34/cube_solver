from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cube.cubies import CubieCube
from cube.kociemba.coords import (
    N_CORNERS,
    N_FLIP,
    N_SLICE,
    N_SLICE_PERM,
    N_TWIST,
    N_UD_EDGES,
    get_corners,
    get_flip,
    get_slice,
    get_slice_perm,
    get_twist,
    get_ud_edges,
    set_corners,
    set_flip,
    set_slice,
    set_slice_perm,
    set_twist,
    set_ud_edges,
)
from cube.moves import MOVE_NAMES

PHASE_1_MOVES = MOVE_NAMES
PHASE_2_MOVES = ("U", "U'", "U2", "D", "D'", "D2", "R2", "L2", "F2", "B2")

DATA_DIR = Path(__file__).parent / "data"


def build_move_table(
    get: Callable[[CubieCube], int],
    set_: Callable[[int], CubieCube],
    size: int,
    moves: tuple[str, ...],
) -> NDArray[np.int32]:
    """table[value, move] = the coordinate after doing moves[m] to a cube with that value. Shape (size, len(moves))."""
    table = np.empty((size, len(moves)), dtype=np.int32)

    for value in range(size):
        cubies = set_(value)

        for m, move in enumerate(moves):
            after = cubies.apply_move(move)
            table[value, m] = get(after)

    return table


def build_pruning_table(
    move_a: NDArray[np.int32], move_b: NDArray[np.int32]
) -> NDArray[np.int8]:
    """Fewest moves from each (a, b) pair to (0, 0), found by breadth-first search.
    Shape (size_a, size_b), indexed prune[a, b]."""
    size_a, size_b = move_a.shape[0], move_b.shape[0]
    prune = np.full((size_a, size_b), -1, dtype=np.int8)
    prune[0, 0] = 0

    depth = 0
    while True:
        found = False

        for value_a in range(size_a):
            for value_b in range(size_b):
                if prune[value_a, value_b] != depth:
                    continue

                for m in range(move_a.shape[1]):
                    next_a = move_a[value_a, m]
                    next_b = move_b[value_b, m]

                    if prune[next_a, next_b] == -1:
                        prune[next_a, next_b] = depth + 1
                        found = True

        if not found:
            break

        depth += 1

    return prune


def _load_or_build(name: str, build: Callable[[], NDArray]) -> NDArray:
    """Load DATA_DIR/name.npy if it exists. Otherwise build it, save it, and return it."""
    path = DATA_DIR / f"{name}.npy"

    if path.exists():
        return np.load(path)

    out = build()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


@dataclass(frozen=True)
class Tables:
    twist_move: NDArray[np.int32]
    flip_move: NDArray[np.int32]
    slice_move: NDArray[np.int32]
    corners_move: NDArray[np.int32]
    ud_edges_move: NDArray[np.int32]
    slice_perm_move: NDArray[np.int32]

    twist_slice_prune: NDArray[np.int8]
    flip_slice_prune: NDArray[np.int8]
    corners_slice_perm_prune: NDArray[np.int8]
    ud_edges_slice_perm_prune: NDArray[np.int8]


@cache
def get_tables() -> Tables:
    """Every table the solver needs, built once."""
    twist_move = _load_or_build(
        "twist_move",
        lambda: build_move_table(get_twist, set_twist, N_TWIST, PHASE_1_MOVES),
    )
    flip_move = _load_or_build(
        "flip_move",
        lambda: build_move_table(get_flip, set_flip, N_FLIP, PHASE_1_MOVES),
    )
    slice_move = _load_or_build(
        "slice_move",
        lambda: build_move_table(get_slice, set_slice, N_SLICE, PHASE_1_MOVES),
    )
    corners_move = _load_or_build(
        "corners_move",
        lambda: build_move_table(get_corners, set_corners, N_CORNERS, PHASE_2_MOVES),
    )
    ud_edges_move = _load_or_build(
        "ud_edges_move",
        lambda: build_move_table(get_ud_edges, set_ud_edges, N_UD_EDGES, PHASE_2_MOVES),
    )
    slice_perm_move = _load_or_build(
        "slice_perm_move",
        lambda: build_move_table(
            get_slice_perm, set_slice_perm, N_SLICE_PERM, PHASE_2_MOVES
        ),
    )

    return Tables(
        twist_move=twist_move,
        flip_move=flip_move,
        slice_move=slice_move,
        corners_move=corners_move,
        ud_edges_move=ud_edges_move,
        slice_perm_move=slice_perm_move,
        twist_slice_prune=_load_or_build(
            "twist_slice_prune",
            lambda: build_pruning_table(twist_move, slice_move),
        ),
        flip_slice_prune=_load_or_build(
            "flip_slice_prune",
            lambda: build_pruning_table(flip_move, slice_move),
        ),
        corners_slice_perm_prune=_load_or_build(
            "corners_slice_perm_prune",
            lambda: build_pruning_table(corners_move, slice_perm_move),
        ),
        ud_edges_slice_perm_prune=_load_or_build(
            "ud_edges_slice_perm_prune",
            lambda: build_pruning_table(ud_edges_move, slice_perm_move),
        ),
    )
