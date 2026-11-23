from itertools import pairwise

import numpy as np
import pytest

from cube.faces import TOTAL_STICKERS, Face
from cube.state import SOLVED_STICKERS, Cube, random_moves

SOLVED_STRING = "".join(face.name * 9 for face in Face)

SOLVED_NET = """\
        U U U
        U U U
        U U U
L L L | F F F | R R R | B B B
L L L | F F F | R R R | B B B
L L L | F F F | R R R | B B B
        D D D
        D D D
        D D D"""


def _solved_with(changes: dict[int, int]) -> np.ndarray:
    stickers = SOLVED_STICKERS.copy()
    for i, value in changes.items():
        stickers[i] = value
    return stickers


def _unsolved_cube() -> Cube:
    return Cube(_solved_with({0: Face.F, 18: Face.U}))


def test_init_accepts_list():
    cube = Cube(list(SOLVED_STICKERS))
    assert cube.is_solved()


def test_init_stores_uint8_with_shape_54():
    cube = Cube(list(SOLVED_STICKERS))
    assert cube.stickers.dtype == np.uint8
    assert cube.stickers.shape == (TOTAL_STICKERS,)


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64],
)
def test_init_accepts_other_integer_dtypes(dtype):
    cube = Cube(SOLVED_STICKERS.astype(dtype))
    assert cube.stickers.dtype == np.uint8
    assert cube.is_solved()


def test_init_copies_input():
    source = SOLVED_STICKERS.copy()
    cube = Cube(source)
    source[0] = Face.B
    assert cube.is_solved()


@pytest.mark.parametrize(
    "stickers",
    [
        SOLVED_STICKERS[:-1],
        np.append(SOLVED_STICKERS, 0),
        [],
        SOLVED_STICKERS.reshape(6, 9),
    ],
)
def test_init_rejects_wrong_shape(stickers):
    with pytest.raises(ValueError, match="Shape"):
        Cube(stickers)


@pytest.mark.parametrize(
    "stickers",
    [SOLVED_STICKERS.astype(float), list(SOLVED_STRING)],
)
def test_init_rejects_non_integer_dtype(stickers):
    with pytest.raises(ValueError, match="integers"):
        Cube(stickers)


@pytest.mark.parametrize(
    "stickers",
    [list(SOLVED_STICKERS[:-1]) + [6], [-1] + list(SOLVED_STICKERS[1:])],
)
def test_init_rejects_out_of_range_values(stickers):
    with pytest.raises(ValueError, match="0-5"):
        Cube(stickers)


def test_stickers_is_read_only():
    cube = Cube.solved()
    with pytest.raises(ValueError):
        cube.stickers[0] = Face.B


def test_solved_is_solved():
    assert Cube.solved().is_solved()


def test_is_solved_false_when_stickers_swapped():
    assert not _unsolved_cube().is_solved()


def test_eq_same_stickers():
    assert Cube(SOLVED_STICKERS) == Cube(list(SOLVED_STICKERS))


def test_eq_different_stickers():
    assert Cube.solved() != _unsolved_cube()


@pytest.mark.parametrize("other", [SOLVED_STRING, SOLVED_STICKERS])
def test_eq_with_non_cube_is_false(other):
    assert (Cube.solved() == other) is False


def test_cubes_work_in_a_set():
    cubes = {Cube.solved(), Cube.solved(), _unsolved_cube()}
    assert len(cubes) == 2


def test_to_string_solved():
    assert Cube.solved().to_string() == SOLVED_STRING


def test_from_string_solved():
    assert Cube.from_string(SOLVED_STRING) == Cube.solved()


def test_string_round_trip():
    cube = _unsolved_cube()
    assert Cube.from_string(cube.to_string()) == cube


@pytest.mark.parametrize(
    "s",
    [SOLVED_STRING[:-1], SOLVED_STRING + "U", ""],
)
def test_from_string_rejects_wrong_length(s):
    with pytest.raises(ValueError, match="letters"):
        Cube.from_string(s)


@pytest.mark.parametrize(
    "s",
    ["X" + SOLVED_STRING[1:], SOLVED_STRING.lower()],
)
def test_from_string_rejects_bad_letters(s):
    with pytest.raises(ValueError):
        Cube.from_string(s)


def test_repr_round_trips():
    cube = _unsolved_cube()
    assert eval(repr(cube)) == cube


def test_str_solved():
    assert str(Cube.solved()) == SOLVED_NET


def test_str_shows_orientation_of_u_and_f():
    lines = str(Cube(_solved_with({0: Face.R, 18: Face.D}))).splitlines()
    assert lines[0] == "        R U U"
    assert lines[3] == "L L L | D F F | R R R | B B B"


def test_to_array_returns_writable_copy():
    cube = Cube.solved()
    array = cube.to_array()
    array[0] = Face.B
    assert cube.is_solved()


@pytest.mark.parametrize("moves", ["", []])
def test_apply_moves_empty_returns_equal_cube(moves):
    cube = _unsolved_cube()
    assert cube.apply_moves(moves) == cube


def test_apply_move_returns_new_cube():
    cube = Cube.solved()
    turned = cube.apply_move("R")
    assert isinstance(turned, Cube)
    assert not turned.is_solved()
    assert cube.is_solved()


def test_apply_move_rejects_unknown_move():
    with pytest.raises(ValueError, match="Unknown move"):
        Cube.solved().apply_move("X")


@pytest.mark.parametrize("moves", ["R U R' U'", ["R", "U", "R'", "U'"]])
def test_apply_moves_accepts_string_or_list(moves):
    assert Cube.solved().apply_moves(moves) == Cube.solved().apply_moves("R U R' U'")


def test_apply_moves_sexy_move_six_times_is_solved():
    assert Cube.solved().apply_moves("R U R' U' " * 6).is_solved()


def test_scramble_is_not_solved():
    assert not Cube.scramble(20, seed=0).is_solved()


def test_scramble_zero_moves_is_solved():
    assert Cube.scramble(0, seed=0).is_solved()


def test_scramble_same_seed_gives_same_cube():
    assert Cube.scramble(20, seed=42) == Cube.scramble(20, seed=42)


def test_scramble_different_seeds_give_different_cubes():
    assert Cube.scramble(20, seed=1) != Cube.scramble(20, seed=2)


def test_random_moves_with_a_seed():
    assert random_moves(10, seed=0) == [
        "B",
        "D2",
        "R'",
        "U",
        "R",
        "L2",
        "D2",
        "B'",
        "D",
        "B2",
    ]


@pytest.mark.parametrize("num_moves", [0, 1, 2, 25, 100])
def test_random_moves_gives_the_number_asked_for(num_moves):
    assert len(random_moves(num_moves, seed=3)) == num_moves


@pytest.mark.parametrize("seed", range(20))
def test_random_moves_never_turns_the_same_face_twice_in_a_row(seed):
    moves = random_moves(50, seed=seed)
    assert all(a[0] != b[0] for a, b in pairwise(moves))


@pytest.mark.parametrize("seed", range(20))
def test_random_moves_never_turns_one_axis_three_times_in_a_row(seed):
    moves = random_moves(50, seed=seed)
    axes = [
        {"U": "UD", "D": "UD", "R": "RL", "L": "RL", "F": "FB", "B": "FB"}[m[0]]
        for m in moves
    ]
    assert all(len({a, b, c}) > 1 for a, b, c in zip(axes, axes[1:], axes[2:]))


@pytest.mark.parametrize("seed", range(5))
def test_scramble_is_the_random_moves_applied(seed):
    moves = random_moves(25, seed=seed)
    assert Cube.scramble(25, seed=seed) == Cube.solved().apply_moves(moves)
