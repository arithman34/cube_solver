import pytest

from cube.cfop.stages import (
    is_cross_solved,
    is_f2l_solved,
    is_oll_solved,
    is_pll_solved,
    solved_pairs,
)
from cube.cubies import CubieCube
from cube.state import Cube

SUNE = "R U R' U R U2 R'"
T_PERM = "R U R' U' R' F R2 U' R' U' R U R' F'"


def _cubies(moves: str) -> CubieCube:
    return CubieCube.from_stickers(Cube.solved().apply_moves(moves).stickers)


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", True),
        ("U", True),
        ("R", False),
        ("D", False),
        ("R2", False),
        ("R U R'", True),
        ("F2 U2 F2", False),
        (SUNE, True),
    ],
)
def test_is_cross_solved(moves, expected):
    assert is_cross_solved(_cubies(moves)) is expected


def test_cross_with_a_flipped_edge_is_not_solved():
    solved = CubieCube.solved()
    edge_orient = solved.edge_orient.copy()
    edge_orient[[0, 5]] = 1
    flipped = CubieCube(
        solved.corner_perm, solved.corner_orient, solved.edge_perm, edge_orient
    )
    assert is_cross_solved(flipped) is False


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", [0, 1, 2, 3]),
        ("U2", [0, 1, 2, 3]),
        ("R", [1, 2]),
        ("F", [2, 3]),
        ("L", [0, 3]),
        ("B", [0, 1]),
        ("R L", []),
        ("D", []),
        ("R U R'", [1, 2, 3]),
        ("F U F'", [0, 2, 3]),
        ("L U L'", [0, 1, 3]),
        ("B U B'", [0, 1, 2]),
        (T_PERM, [0, 1, 2, 3]),
    ],
)
def test_solved_pairs(moves, expected):
    assert solved_pairs(_cubies(moves)) == expected


def test_twisted_corner_does_not_count_as_a_pair():
    solved = CubieCube.solved()
    corner_orient = solved.corner_orient.copy()
    corner_orient[[0, 6]] = [1, 2]
    twisted = CubieCube(
        solved.corner_perm, corner_orient, solved.edge_perm, solved.edge_orient
    )
    assert solved_pairs(twisted) == [0, 1, 3]


@pytest.mark.parametrize(
    "moves, expected",
    [("", True), ("U'", True), ("R U R'", False), ("D", False), (SUNE, True)],
)
def test_is_f2l_solved(moves, expected):
    assert is_f2l_solved(_cubies(moves)) is expected


def test_four_pairs_without_the_cross_is_not_f2l():
    solved = CubieCube.solved()
    edge_perm = solved.edge_perm.copy()
    edge_perm[[4, 5, 6]] = edge_perm[[5, 6, 4]]
    cubies = CubieCube(
        solved.corner_perm, solved.corner_orient, edge_perm, solved.edge_orient
    )
    assert solved_pairs(cubies) == [0, 1, 2, 3]
    assert is_f2l_solved(cubies) is False


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", True),
        ("U", True),
        (SUNE, False),
        ("F R U R' U' F'", False),
        (T_PERM, True),
        ("R U R'", False),
        ("D", False),
    ],
)
def test_is_oll_solved(moves, expected):
    assert is_oll_solved(_cubies(moves)) is expected


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", True),
        ("U", False),
        (SUNE, False),
        (T_PERM, False),
        (f"{T_PERM} {T_PERM}", True),
    ],
)
def test_is_pll_solved(moves, expected):
    assert is_pll_solved(_cubies(moves)) is expected


@pytest.mark.parametrize("seed", range(5))
def test_scrambled_cube_has_no_stage_done(seed):
    cubies = CubieCube.from_stickers(Cube.scramble(30, seed=seed).stickers)
    assert not is_cross_solved(cubies)
    assert not is_f2l_solved(cubies)
    assert not is_oll_solved(cubies)
    assert not is_pll_solved(cubies)
