from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from cube.moves import MOVE_NAMES
from cube.state import SOLVED_STICKERS, Cube
from training.data import IDENTITY, MOVE_PERMS
from training.evaluate import predict
from training.model import ValueNet


@dataclass
class SearchResult:
    solved: bool
    moves: list[str]
    nodes: int  # how many cubes the model was asked about


def neighbours(stickers: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """The 18 cubes one move away, shape (18, 54), in MOVE_NAMES order."""
    return stickers[MOVE_PERMS[:IDENTITY]]


def greedy_search(
    model: ValueNet, cube: Cube, device: torch.device, max_steps: int = 50
) -> SearchResult:
    """Repeatedly make the move whose result the model rates closest to solved."""
    stickers = cube.stickers
    moves, visited = [], {stickers.tobytes()}
    nodes = 0
    for _ in range(max_steps):
        if (stickers == SOLVED_STICKERS).all():
            return SearchResult(solved=True, moves=moves, nodes=nodes)

        next_cubes = neighbours(stickers)
        solved = (next_cubes == SOLVED_STICKERS).all(axis=1)
        if solved.any():
            move = MOVE_NAMES[solved.argmax()]
            return SearchResult(solved=True, moves=moves + [move], nodes=nodes)

        values = predict(model, next_cubes, device)
        nodes += len(next_cubes)
        for i, next_cube in enumerate(next_cubes):
            if next_cube.tobytes() in visited:
                values[i] = np.inf

        best = values.argmin()
        if values[best] == np.inf:
            return SearchResult(solved=False, moves=moves, nodes=nodes)

        stickers = next_cubes[best]
        moves.append(MOVE_NAMES[best])
        visited.add(stickers.tobytes())

    return SearchResult(solved=False, moves=moves, nodes=nodes)
