from enum import Enum

import numpy as np

from cube.cubies import CubieCube, permutation_parity
from cube.faces import NUM_FACES, NUM_STICKERS_PER_FACE, Face, center_index
from cube.state import Cube


class CubeError(Enum):
    """A reason a cube can't be solved."""

    BAD_CENTERS = "Centers are not in the expected places."
    BAD_COUNTS = "Each colour must appear exactly 9 times."
    INVALID_PIECE = (
        "A piece has a colour combination that doesn't exist, or appears twice."
    )
    TWISTED_CORNER = "A corner is twisted."
    FLIPPED_EDGE = "An edge is flipped."
    SWAPPED_PIECES = "Two pieces are swapped."


def _check_centers(cube: Cube) -> bool:
    """True if every center holds its own face."""
    return all(cube.stickers[center_index(face)] == face for face in Face)


def _check_counts(cube: Cube) -> bool:
    """True if every face value appears exactly 9 times."""
    counts = np.bincount(cube.stickers, minlength=NUM_FACES)
    return bool(np.all(counts == NUM_STICKERS_PER_FACE))


def find_errors(cube: Cube) -> list[CubeError]:
    """Every reason the cube can't be solved; empty if it can."""
    if not _check_centers(cube):
        return [CubeError.BAD_CENTERS]
    if not _check_counts(cube):
        return [CubeError.BAD_COUNTS]

    try:
        cubies = CubieCube.from_stickers(cube.stickers)
    except ValueError:
        return [CubeError.INVALID_PIECE]

    errors = []
    if cubies.corner_orient.sum() % 3 != 0:
        errors.append(CubeError.TWISTED_CORNER)
    if cubies.edge_orient.sum() % 2 != 0:
        errors.append(CubeError.FLIPPED_EDGE)
    if permutation_parity(cubies.corner_perm) != permutation_parity(cubies.edge_perm):
        errors.append(CubeError.SWAPPED_PIECES)
    return errors


def is_solvable(cube: Cube) -> bool:
    """True if find_errors finds nothing."""
    return not find_errors(cube)
