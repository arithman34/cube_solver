import numpy as np
import pytest

from cube.cubies import (
    CORNER_FACELETS,
    EDGE_FACELETS,
    NUM_CORNERS,
    NUM_EDGES,
    CubieCube,
    permutation_parity,
)
from cube.faces import Face, center_index
from cube.state import SOLVED_STICKERS, Cube


def _cubies(moves: str) -> CubieCube:
    return CubieCube.from_stickers(Cube.solved().apply_moves(moves).stickers)


def test_facelets_cover_every_non_center_sticker_once():
    used = sorted(CORNER_FACELETS.flatten().tolist() + EDGE_FACELETS.flatten().tolist())
    centers = {center_index(face) for face in Face}
    assert used == [i for i in range(54) if i not in centers]


def test_solved():
    cubies = CubieCube.solved()
    assert cubies.corner_perm.tolist() == list(range(NUM_CORNERS))
    assert cubies.corner_orient.tolist() == [0] * NUM_CORNERS
    assert cubies.edge_perm.tolist() == list(range(NUM_EDGES))
    assert cubies.edge_orient.tolist() == [0] * NUM_EDGES


def test_from_stickers_solved():
    cubies = CubieCube.from_stickers(SOLVED_STICKERS)
    solved = CubieCube.solved()
    assert np.array_equal(cubies.corner_perm, solved.corner_perm)
    assert np.array_equal(cubies.corner_orient, solved.corner_orient)
    assert np.array_equal(cubies.edge_perm, solved.edge_perm)
    assert np.array_equal(cubies.edge_orient, solved.edge_orient)


def test_from_stickers_after_r():
    cubies = _cubies("R")
    assert cubies.corner_perm.tolist() == [4, 1, 2, 0, 7, 5, 6, 3]
    assert cubies.corner_orient.tolist() == [2, 0, 0, 1, 1, 0, 0, 2]
    assert cubies.edge_perm.tolist() == [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0]
    assert cubies.edge_orient.tolist() == [0] * NUM_EDGES


def test_from_stickers_after_f():
    cubies = _cubies("F")
    assert cubies.corner_perm.tolist() == [1, 5, 2, 3, 0, 4, 6, 7]
    assert cubies.corner_orient.tolist() == [1, 2, 0, 0, 2, 1, 0, 0]
    assert cubies.edge_perm.tolist() == [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11]
    assert cubies.edge_orient.tolist() == [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]


def test_from_stickers_rejects_piece_that_does_not_exist():
    stickers = SOLVED_STICKERS.copy()
    stickers[[8, 10]] = stickers[[10, 8]]
    with pytest.raises(ValueError):
        CubieCube.from_stickers(stickers)


def test_from_stickers_rejects_duplicate_piece():
    stickers = SOLVED_STICKERS.copy()
    stickers[CORNER_FACELETS[1]] = stickers[CORNER_FACELETS[0]]
    with pytest.raises(ValueError):
        CubieCube.from_stickers(stickers)


@pytest.mark.parametrize(
    "perm, expected",
    [
        ([0, 1, 2, 3], 0),
        ([1, 0, 2, 3], 1),
        ([1, 2, 0, 3], 0),
        ([1, 2, 3, 0], 1),
        ([1, 0, 3, 2], 0),
    ],
)
def test_permutation_parity(perm, expected):
    assert permutation_parity(perm) == expected


@pytest.mark.parametrize("move", [face.name for face in Face])
def test_quarter_turn_has_odd_corner_and_edge_parity(move):
    cubies = _cubies(move)
    assert permutation_parity(cubies.corner_perm) == 1
    assert permutation_parity(cubies.edge_perm) == 1


@pytest.mark.parametrize("seed", range(5))
def test_scrambled_cube_keeps_the_rules(seed):
    cube = Cube.scramble(30, seed=seed)
    cubies = CubieCube.from_stickers(cube.stickers)
    assert cubies.corner_orient.sum() % 3 == 0
    assert cubies.edge_orient.sum() % 2 == 0
    assert permutation_parity(cubies.corner_perm) == permutation_parity(cubies.edge_perm)
