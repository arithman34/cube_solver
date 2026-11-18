import numpy as np
from numpy.typing import NDArray

from cube.faces import FACE_SIZE, TOTAL_STICKERS, Face, index


def _face_grid(face: Face) -> NDArray[np.intp]:
    """The 9 flat indices of a face, on a 3x3 grid."""
    return np.array(
        [
            [index(face, row, col) for col in range(FACE_SIZE)]
            for row in range(FACE_SIZE)
        ]
    )


def _row(face: Face, row: int, reverse: bool = False) -> list[int]:
    """The 3 indices along one row of a face, left to right."""
    strip = [index(face, row, col) for col in range(FACE_SIZE)]
    return strip[::-1] if reverse else strip


def _col(face: Face, col: int, reverse: bool = False) -> list[int]:
    """The 3 indices down one column of a face, top to bottom."""
    strip = [index(face, row, col) for row in range(FACE_SIZE)]
    return strip[::-1] if reverse else strip


_SIDE_STRIPS: dict[Face, list[list[int]]] = {
    Face.U: [_row(Face.F, 0), _row(Face.L, 0), _row(Face.B, 0), _row(Face.R, 0)],
    Face.R: [
        _col(Face.U, 2, reverse=True),
        _col(Face.B, 0),
        _col(Face.D, 2, reverse=True),
        _col(Face.F, 2, reverse=True),
    ],
    Face.F: [
        _col(Face.L, 2, reverse=True),
        _row(Face.U, 2),
        _col(Face.R, 0),
        _row(Face.D, 0, reverse=True),
    ],
    Face.D: [_row(Face.R, 2), _row(Face.B, 2), _row(Face.L, 2), _row(Face.F, 2)],
    Face.L: [
        _col(Face.F, 0, reverse=True),
        _col(Face.D, 0, reverse=True),
        _col(Face.B, 2),
        _col(Face.U, 0, reverse=True),
    ],
    Face.B: [
        _row(Face.D, 2, reverse=True),
        _col(Face.R, 2),
        _row(Face.U, 0),
        _col(Face.L, 0, reverse=True),
    ],
}


def _quarter_turn(face: Face) -> NDArray[np.intp]:
    """Build the clockwise quarter-turn permutation for one face."""
    perm = np.arange(TOTAL_STICKERS)

    grid = _face_grid(face)
    rotated = np.rot90(grid, k=-1)
    perm[grid.flatten()] = rotated.flatten()

    strips = _SIDE_STRIPS[face]
    for src, dst in zip(strips, strips[1:] + strips[:1]):
        perm[dst] = src

    return perm


def _inverse(perm: NDArray[np.intp]) -> NDArray[np.intp]:
    """The permutation that undoes perm."""
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    return inverse


def _compose(first: NDArray[np.intp], second: NDArray[np.intp]) -> NDArray[np.intp]:
    """One permutation equal to applying first, then second."""
    return first[second]


def _build_moves() -> dict[str, NDArray[np.intp]]:
    """Build the 18 quarter-turn moves and their inverses and doubles."""
    out: dict[str, NDArray[np.intp]] = {}
    for face in Face:
        turn = _quarter_turn(face)
        inverse_turn = _inverse(turn)
        double_turn = _compose(turn, turn)

        turn.flags.writeable = False
        inverse_turn.flags.writeable = False
        double_turn.flags.writeable = False

        out[face.name] = turn
        out[face.name + "'"] = inverse_turn
        out[face.name + "2"] = double_turn

    return out


MOVES = _build_moves()

MOVE_NAMES = tuple(MOVES)


def parse_moves(text: str) -> list[str]:
    """Split "R U R' U2" into ["R", "U", "R'", "U2"]; ValueError on unknown moves."""
    moves = text.split()

    for move in moves:
        if move not in MOVES:
            raise ValueError(f"Unknown move. Got {move!r}.")

    return moves


def invert_moves(moves: list[str]) -> list[str]:
    """The sequence that undoes the moves."""
    moves = moves[::-1]

    for i, move in enumerate(moves):
        suffix = move[1] if len(move) > 1 else ""

        if suffix == "2":
            continue

        base_letter = move[0]
        suffix = "'" if suffix == "" else ""
        moves[i] = base_letter + suffix

    return moves


def apply_move(stickers: NDArray[np.uint8], move: str) -> NDArray[np.uint8]:
    """Apply one move to a (54,) array or an (N, 54) batch."""
    if move not in MOVES:
        raise ValueError(f"Unknown move. Got {move!r}.")

    perm = MOVES[move]

    if stickers.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array. Got {stickers.ndim}D.")

    return stickers[..., perm]
