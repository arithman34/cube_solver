from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from cube.faces import TOTAL_STICKERS
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


def expand(beam: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Every cube one move from a (k, 54) beam, shape (k * 18, 54).

    Child i comes from beam[i // 18] by the move MOVE_NAMES[i % 18].
    """
    return beam[:, MOVE_PERMS[:IDENTITY]].reshape(-1, TOTAL_STICKERS)


def beam_search(
    model: ValueNet,
    cube: Cube,
    device: torch.device,
    beam_width: int = 100,
    max_steps: int = 50,
) -> SearchResult:
    """Keep the beam_width cubes the model rates closest to solved at every step."""
    if cube.is_solved():
        return SearchResult(solved=True, moves=[], nodes=0)

    beam = cube.to_array().reshape(1, -1)
    paths = [[]]
    visited = {cube.stickers.tobytes()}
    nodes = 0

    for _ in range(max_steps):
        children = expand(beam)
        parents = np.arange(len(children)) // len(MOVE_NAMES)
        move_ids = np.arange(len(children)) % len(MOVE_NAMES)

        solved = (children == SOLVED_STICKERS).all(axis=1)
        if solved.any():
            i = solved.argmax()
            moves = paths[parents[i]] + [MOVE_NAMES[move_ids[i]]]
            return SearchResult(solved=True, moves=moves, nodes=nodes)

        keep = np.zeros(len(children), dtype=bool)
        for i, child in enumerate(children):
            key = child.tobytes()
            if key not in visited:
                visited.add(key)
                keep[i] = True

        children, parents, move_ids = children[keep], parents[keep], move_ids[keep]

        if len(children) == 0:
            return SearchResult(solved=False, moves=paths[0], nodes=nodes)

        values = predict(model, children, device)
        nodes += len(children)
        best = np.argsort(values)[:beam_width]

        beam = children[best]
        paths = [paths[parents[i]] + [MOVE_NAMES[move_ids[i]]] for i in best]

    return SearchResult(solved=False, moves=paths[0], nodes=nodes)
