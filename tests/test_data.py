from itertools import pairwise

import numpy as np
import pytest

from cube.moves import MOVE_NAMES
from cube.state import Cube
from training.data import generate, scramble_moves


def test_scramble_moves_never_turns_same_face_twice_in_a_row():
    moves = scramble_moves(500, np.random.default_rng(0))

    assert len(moves) == 500
    assert all(move in MOVE_NAMES for move in moves)
    for previous, current in pairwise(moves):
        assert previous[0] != current[0]


def test_scramble_moves_with_zero_moves():
    assert scramble_moves(0, np.random.default_rng(0)) == []


def test_generate_returns_matching_cubes_and_depths():
    cubes, depths = generate(200, max_moves=8, seed=1)

    assert len(cubes) == len(depths) == 200
    assert all(len(cube) == 54 for cube in cubes)
    assert all(1 <= depth <= 8 for depth in depths)


def test_generate_uses_every_depth():
    _, depths = generate(500, max_moves=5, seed=2)

    assert set(depths) == {1, 2, 3, 4, 5}


def test_generate_single_move_cubes():
    one_move_cubes = {Cube.solved().apply_move(move).to_string() for move in MOVE_NAMES}

    cubes, depths = generate(50, max_moves=1, seed=3)

    assert depths == [1] * 50
    assert set(cubes) <= one_move_cubes


def test_generate_short_scrambles_are_never_solved():
    solved = Cube.solved().to_string()

    cubes, _ = generate(300, max_moves=2, seed=4)

    assert solved not in cubes


def test_generate_is_repeatable_with_a_seed():
    assert generate(20, max_moves=10, seed=5) == generate(20, max_moves=10, seed=5)
    assert generate(20, max_moves=10, seed=5) != generate(20, max_moves=10, seed=6)


def test_generate_zero_examples():
    assert generate(0, max_moves=5, seed=0) == ([], [])


def test_generate_rejects_max_moves_below_one():
    with pytest.raises(ValueError):
        generate(10, max_moves=0)
