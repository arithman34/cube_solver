from itertools import pairwise

import pytest

from cube.cfop import solve
from cube.state import Cube

SUNE = "R U R' U R U2 R'"
T_PERM = "R U R' U' R' F R2 U' R' U' R U R' F'"


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", []),
        ("R", ["R'"]),
        ("U", ["U'"]),
        ("D2", ["D2"]),
        ("F R", ["R'", "F'"]),
        ("R U R' U'", ["U", "R", "U'", "R'"]),
        (T_PERM, T_PERM.split()),
    ],
)
def test_solve_short_scrambles(moves, expected):
    assert solve(Cube.solved().apply_moves(moves)) == expected


def test_solve_sune_with_an_oll_algorithm():
    moves = solve(Cube.solved().apply_moves(SUNE))
    assert moves == ["R", "U2", "R'", "U'", "R", "U'", "R'"]


@pytest.mark.parametrize("seed", range(5))
def test_solve_scrambled_cube(seed):
    cube = Cube.scramble(30, seed=seed)
    moves = solve(cube)
    assert cube.apply_moves(moves).is_solved()
    assert all(a[0] != b[0] for a, b in pairwise(moves))


def test_solve_ignores_the_time_limit():
    cube = Cube.scramble(30, seed=0)
    assert solve(cube, 0.0) == solve(cube, 5.0)


def test_solve_rejects_unsolvable_cube():
    stickers = Cube.solved().to_array()
    stickers[[5, 10]] = stickers[[10, 5]]
    with pytest.raises(ValueError, match="not solvable"):
        solve(Cube(stickers))
