import numpy as np
import pytest

from cube.cubies import (
    CORNER_FACELETS,
    EDGE_FACELETS,
    MOVE_CUBIES,
    NUM_CORNERS,
    NUM_EDGES,
    CubieCube,
    permutation_parity,
)
from cube.faces import Face, center_index
from cube.moves import MOVE_NAMES
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
    assert CubieCube.from_stickers(SOLVED_STICKERS) == CubieCube.solved()


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


def test_eq_different_cubes():
    assert CubieCube.solved() != _cubies("R")


def test_eq_with_non_cubie_cube_is_false():
    assert (CubieCube.solved() == SOLVED_STICKERS) is False


def test_move_cubies_has_all_18_moves():
    assert list(MOVE_CUBIES) == list(MOVE_NAMES)


def test_multiply_by_solved_is_unchanged():
    cubies = _cubies("R U F'")
    assert cubies.multiply(CubieCube.solved()) == cubies
    assert CubieCube.solved().multiply(cubies) == cubies


def test_multiply_matches_sticker_moves():
    assert _cubies("R").multiply(_cubies("F")) == _cubies("R F")


@pytest.mark.parametrize("move", MOVE_NAMES)
def test_apply_move_matches_sticker_move(move):
    assert CubieCube.solved().apply_move(move) == _cubies(move)


@pytest.mark.parametrize("seed", range(3))
def test_apply_move_matches_stickers_after_scramble(seed):
    rng = np.random.default_rng(seed)
    sequence = [str(move) for move in rng.choice(MOVE_NAMES, size=40)]
    cubies = CubieCube.solved()
    for move in sequence:
        cubies = cubies.apply_move(move)
    assert cubies == _cubies(" ".join(sequence))


@pytest.mark.parametrize("face", [face.name for face in Face])
def test_move_then_prime_is_solved(face):
    assert CubieCube.solved().apply_move(face).apply_move(face + "'") == CubieCube.solved()


def test_apply_move_rejects_unknown_move():
    with pytest.raises(ValueError, match="Unknown move"):
        CubieCube.solved().apply_move("X")
