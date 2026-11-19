from itertools import pairwise

import numpy as np
import pytest

from cube.moves import MOVE_NAMES
from cube.state import SOLVED_STICKERS, Cube
from training.data import IDENTITY, apply_scrambles, generate, scramble_moves


def test_scramble_moves_pads_each_row_to_its_depth():
    depths = np.array([1, 3, 5])

    moves = scramble_moves(depths, max_moves=5, rng=np.random.default_rng(0))

    assert moves.shape == (3, 5)
    assert ((moves != IDENTITY).sum(axis=1) == depths).all()
    assert (moves[0, 1:] == IDENTITY).all()
    assert (moves[1, 3:] == IDENTITY).all()


def test_scramble_moves_never_turns_same_face_twice_in_a_row():
    depths = np.full(200, 20)

    moves = scramble_moves(depths, max_moves=20, rng=np.random.default_rng(0))

    for row in moves:
        names = [MOVE_NAMES[i] for i in row]
        for previous, current in pairwise(names):
            assert previous[0] != current[0]


def test_apply_scrambles_matches_cube_moves():
    def index(name):
        return MOVE_NAMES.index(name)

    moves = np.array(
        [
            [index("R"), index("U'"), index("F2")],
            [index("L"), IDENTITY, IDENTITY],
        ]
    )

    stickers = apply_scrambles(moves)

    assert (stickers[0] == Cube.solved().apply_moves("R U' F2").stickers).all()
    assert (stickers[1] == Cube.solved().apply_moves("L").stickers).all()


def test_generate_returns_matching_stickers_and_depths():
    stickers, depths = generate(200, max_moves=8, seed=1)

    assert stickers.shape == (200, 54)
    assert stickers.dtype == np.uint8
    assert depths.shape == (200,)
    assert ((depths >= 1) & (depths <= 8)).all()


def test_generate_uses_every_depth():
    _, depths = generate(500, max_moves=5, seed=2)

    assert set(depths.tolist()) == {1, 2, 3, 4, 5}


def test_generate_single_move_cubes():
    one_move_cubes = {
        Cube.solved().apply_move(move).stickers.tobytes() for move in MOVE_NAMES
    }

    stickers, depths = generate(50, max_moves=1, seed=3)

    assert (depths == 1).all()
    assert {row.tobytes() for row in stickers} <= one_move_cubes


def test_generate_short_scrambles_are_never_solved():
    stickers, _ = generate(300, max_moves=2, seed=4)

    assert not (stickers == SOLVED_STICKERS).all(axis=1).any()


def test_generate_is_repeatable_with_a_seed():
    first_stickers, first_depths = generate(20, max_moves=10, seed=5)
    second_stickers, second_depths = generate(20, max_moves=10, seed=5)
    other_stickers, _ = generate(20, max_moves=10, seed=6)

    assert (first_stickers == second_stickers).all()
    assert (first_depths == second_depths).all()
    assert not (first_stickers == other_stickers).all()


def test_generate_zero_examples():
    stickers, depths = generate(0, max_moves=5, seed=0)

    assert stickers.shape == (0, 54)
    assert depths.shape == (0,)


def test_generate_rejects_max_moves_below_one():
    with pytest.raises(ValueError):
        generate(10, max_moves=0)
