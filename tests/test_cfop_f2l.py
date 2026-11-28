from collections import Counter
from itertools import pairwise

import pytest

from cube.cfop.cross import solve_cross
from cube.cfop.f2l import (
    _corner_moves,
    _pair_distances,
    _pair_key,
    solve_f2l,
    solve_pair,
)
from cube.cfop.stages import is_cross_solved, is_f2l_solved, solved_pairs
from cube.cubies import CubieCube
from cube.kociemba.search import _allowed
from cube.moves import MOVE_NAMES
from cube.state import Cube

SEXY_PAIR_0 = "R U R'"
SEXY_PAIR_1 = "L' U' L"


def _cubies(moves: str) -> CubieCube:
    return CubieCube.from_stickers(Cube.solved().apply_moves(moves).stickers)


def _after(cubies: CubieCube, moves: list[str]) -> CubieCube:
    for move in moves:
        cubies = cubies.apply_move(move)
    return cubies


@pytest.mark.parametrize(
    "move, position, expected",
    [
        ("R", 12, 2),
        ("U", 12, 12),
        ("F", 12, 16),
        ("D", 12, 21),
        ("R", 21, 13),
        ("D", 21, 18),
        ("B", 21, 11),
    ],
)
def test_corner_moves(move, position, expected):
    assert _corner_moves()[move][position] == expected


@pytest.mark.parametrize("move", MOVE_NAMES)
def test_corner_moves_agree_with_turning_the_cube(move):
    cubies = _cubies("R U F D2 L' B")
    for pair in range(4):
        corner, _ = _pair_key(cubies, pair)
        moved, _ = _pair_key(cubies.apply_move(move), pair)
        assert _corner_moves()[move][corner] == moved


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", [(12, 16), (15, 18), (18, 20), (21, 22)]),
        ("U", [(12, 16), (15, 18), (18, 20), (21, 22)]),
        ("R", [(2, 0), (15, 18), (18, 20), (13, 8)]),
        ("F", [(16, 11), (5, 3), (18, 20), (21, 22)]),
        ("D", [(21, 16), (12, 18), (15, 20), (18, 22)]),
        (SEXY_PAIR_0, [(5, 2), (15, 18), (18, 20), (21, 22)]),
    ],
)
def test_pair_key(moves, expected):
    cubies = _cubies(moves)
    assert [_pair_key(cubies, pair) for pair in range(4)] == expected


@pytest.mark.parametrize("pair", range(4))
def test_pair_distances_cover_every_position(pair):
    assert Counter(_pair_distances(pair).values()) == {
        0: 1,
        1: 9,
        2: 54,
        3: 195,
        4: 265,
        5: 50,
        6: 2,
    }


@pytest.mark.parametrize(
    "moves, pair, expected",
    [
        ("", 0, []),
        (SEXY_PAIR_0, 0, ["R", "U'", "R'"]),
        ("F' U' F", 0, ["F'", "U", "F"]),
        ("R U R' U'", 0, ["U", "R", "U'", "R'"]),
        (SEXY_PAIR_1, 1, ["L'", "U", "L"]),
    ],
)
def test_solve_pair(moves, pair, expected):
    assert solve_pair(_cubies(moves), pair, []) == expected


def test_solve_pair_keeps_the_pairs_already_done():
    cubies = _cubies("R U R' F' U' F")
    assert solved_pairs(cubies) == [1, 2, 3]

    moves = solve_pair(cubies, 0, [1])
    assert moves == ["R", "U'", "R'"]
    assert solved_pairs(_after(cubies, moves)) == [0, 1, 2, 3]


def test_solve_pair_does_not_start_on_the_previous_face():
    assert solve_pair(_cubies(SEXY_PAIR_0), 0, [], previous="R") == [
        "U",
        "R",
        "U2",
        "R'",
    ]


def test_solve_f2l_already_solved():
    assert solve_f2l(_cubies("R U R' U R U2 R'")) == []


def test_solve_f2l_two_pairs():
    assert solve_f2l(_cubies(f"{SEXY_PAIR_0} {SEXY_PAIR_1}")) == [
        "L'",
        "U",
        "L",
        "U",
        "R",
        "U2",
        "R'",
    ]


@pytest.mark.parametrize("seed", range(3))
def test_cross_then_f2l_on_a_scrambled_cube(seed):
    cubies = CubieCube.from_stickers(Cube.scramble(30, seed=seed).stickers)
    cross = solve_cross(cubies)
    f2l = solve_f2l(_after(cubies, cross), previous=cross[-1])

    solved = _after(cubies, cross + f2l)
    assert is_cross_solved(solved)
    assert is_f2l_solved(solved)
    assert all(_allowed(a, b) for a, b in pairwise(cross + f2l))
