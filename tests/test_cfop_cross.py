from collections import Counter

import pytest

from cube.cfop.cross import (
    _cross_key,
    _distances,
    _edge_moves,
    _move_key,
    solve_cross,
)
from cube.cfop.stages import is_cross_solved
from cube.cubies import CubieCube
from cube.moves import MOVE_NAMES
from cube.state import Cube


def _cubies(moves: str) -> CubieCube:
    return CubieCube.from_stickers(Cube.solved().apply_moves(moves).stickers)


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", (8, 10, 12, 14)),
        ("U", (8, 10, 12, 14)),
        ("R", (16, 10, 12, 14)),
        ("F", (8, 19, 12, 14)),
        ("F2", (8, 2, 12, 14)),
        ("D", (14, 8, 10, 12)),
        ("R U F", (11, 19, 12, 14)),
    ],
)
def test_cross_key(moves, expected):
    assert _cross_key(_cubies(moves)) == expected


def test_cross_key_leaves_the_cube_alone():
    cubies = _cubies("R U F")
    _cross_key(cubies)
    assert cubies == _cubies("R U F")


@pytest.mark.parametrize(
    "move, position, expected",
    [
        ("R", 8, 16),
        ("R", 9, 17),
        ("R", 10, 10),
        ("F", 10, 19),
        ("F", 11, 18),
        ("F2", 10, 2),
        ("D", 8, 14),
        ("U", 14, 14),
    ],
)
def test_edge_moves(move, position, expected):
    assert _edge_moves()[move][position] == expected


@pytest.mark.parametrize("move", MOVE_NAMES)
def test_edge_moves_agree_with_turning_the_cube(move):
    cubies = _cubies("R U F D2 L' B")
    moved = _move_key(_cross_key(cubies), _edge_moves()[move])
    assert moved == _cross_key(cubies.apply_move(move))


def test_distances_cover_every_cross():
    assert Counter(_distances().values()) == {
        0: 1,
        1: 15,
        2: 158,
        3: 1394,
        4: 9809,
        5: 46381,
        6: 97254,
        7: 34966,
        8: 102,
    }


@pytest.mark.parametrize("moves", ["", "U", "U2", "R U R' U'"])
def test_solve_cross_already_solved(moves):
    assert solve_cross(_cubies(moves)) == []


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("R", ["R'"]),
        ("F'", ["F"]),
        ("D2", ["D2"]),
        ("R U", ["R'"]),
        ("R U F", ["F'", "R'"]),
        ("F R B L", ["L'", "B'", "R'", "F'"]),
    ],
)
def test_solve_cross_short_scrambles(moves, expected):
    assert solve_cross(_cubies(moves)) == expected


def test_solve_cross_fixes_a_flipped_edge():
    solved = CubieCube.solved()
    edge_orient = solved.edge_orient.copy()
    edge_orient[[0, 5]] = 1
    cubies = CubieCube(
        solved.corner_perm, solved.corner_orient, solved.edge_perm, edge_orient
    )

    moves = solve_cross(cubies)
    for move in moves:
        cubies = cubies.apply_move(move)
    assert moves
    assert is_cross_solved(cubies)


@pytest.mark.parametrize("seed", range(20))
def test_solve_cross_scrambled_cube(seed):
    cube = Cube.scramble(30, seed=seed)
    moves = solve_cross(CubieCube.from_stickers(cube.stickers))
    solved = cube.apply_moves(moves)
    assert is_cross_solved(CubieCube.from_stickers(solved.stickers))
    assert len(moves) <= 8
