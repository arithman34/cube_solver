import numpy as np
import pytest

from cube.faces import TOTAL_STICKERS, Face
from cube.state import SOLVED_STICKERS, Cube

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


@pytest.mark.parametrize(
    "other",
    [SOLVED_STRING, SOLVED_STICKERS]
)
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
