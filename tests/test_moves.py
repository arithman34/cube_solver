import numpy as np
import pytest

from cube.faces import TOTAL_STICKERS, Face, center_index
from cube.moves import MOVE_NAMES, MOVES, apply_move, invert_moves, parse_moves
from cube.state import LETTERS, SOLVED_STICKERS

IDENTITY = np.arange(TOTAL_STICKERS)
BASE_FACES = [face.name for face in Face]


def _apply(stickers: np.ndarray, moves: list[str]) -> np.ndarray:
    for move in moves:
        stickers = apply_move(stickers, move)
    return stickers


def _to_string(stickers: np.ndarray) -> str:
    return "".join(LETTERS[stickers])


def test_move_names():
    assert MOVE_NAMES == (
        "U", "U'", "U2",
        "R", "R'", "R2",
        "F", "F'", "F2",
        "D", "D'", "D2",
        "L", "L'", "L2",
        "B", "B'", "B2",
    )  # fmt: skip


@pytest.mark.parametrize("move", MOVE_NAMES)
def test_every_move_is_a_permutation(move):
    assert np.array_equal(np.sort(MOVES[move]), IDENTITY)


@pytest.mark.parametrize("face", BASE_FACES)
def test_quarter_turn_moves_20_stickers_and_keeps_centers(face):
    perm = MOVES[face]
    assert np.count_nonzero(perm != IDENTITY) == 20
    for center_face in Face:
        assert perm[center_index(center_face)] == center_index(center_face)


@pytest.mark.parametrize("face", BASE_FACES)
def test_quarter_turn_four_times_is_identity(face):
    assert np.array_equal(_apply(SOLVED_STICKERS, [face] * 4), SOLVED_STICKERS)


@pytest.mark.parametrize("face", BASE_FACES)
def test_move_then_prime_is_identity(face):
    scrambled = _apply(SOLVED_STICKERS, ["R", "U", "F"])
    after = _apply(scrambled, [face, face + "'"])
    assert np.array_equal(after, scrambled)


@pytest.mark.parametrize("face", BASE_FACES)
def test_double_turn_equals_two_quarter_turns(face):
    assert np.array_equal(
        apply_move(SOLVED_STICKERS, face + "2"),
        _apply(SOLVED_STICKERS, [face, face]),
    )


@pytest.mark.parametrize(
    "move, expected",
    [
        ("U", "UUUUUUUUUBBBRRRRRRRRRFFFFFFDDDDDDDDDFFFLLLLLLLLLBBBBBB"),
        ("R", "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"),
        ("F", "UUUUUULLLURRURRURRFFFFFFFFFRRRDDDDDDLLDLLDLLDBBBBBBBBB"),
    ],
)
def test_move_on_solved_cube(move, expected):
    assert _to_string(apply_move(SOLVED_STICKERS, move)) == expected


def test_sexy_move_six_times_is_identity():
    """Sexy move is R U R' U'."""
    assert np.array_equal(
        _apply(SOLVED_STICKERS, ["R", "U", "R'", "U'"] * 6), SOLVED_STICKERS
    )


def test_r_u_has_order_105():
    stickers = SOLVED_STICKERS
    solved_after = []
    for repeats in range(1, 106):
        stickers = _apply(stickers, ["R", "U"])
        if np.array_equal(stickers, SOLVED_STICKERS):
            solved_after.append(repeats)
    assert solved_after == [105]


def test_moves_are_read_only():
    with pytest.raises(ValueError):
        MOVES["R"][0] = 1


def test_apply_move_does_not_modify_input():
    stickers = SOLVED_STICKERS.copy()
    apply_move(stickers, "R")
    assert np.array_equal(stickers, SOLVED_STICKERS)


def test_apply_move_on_batch():
    batch = np.stack([SOLVED_STICKERS, apply_move(SOLVED_STICKERS, "U")])
    result = apply_move(batch, "R")
    assert result.shape == (2, TOTAL_STICKERS)
    assert np.array_equal(result[0], apply_move(SOLVED_STICKERS, "R"))
    assert np.array_equal(result[1], _apply(SOLVED_STICKERS, ["U", "R"]))


@pytest.mark.parametrize("move", ["X", "r", "R3", ""])
def test_apply_move_rejects_unknown_move(move):
    with pytest.raises(ValueError):
        apply_move(SOLVED_STICKERS, move)


def test_parse_moves():
    assert parse_moves("R U R' U2") == ["R", "U", "R'", "U2"]


def test_parse_moves_ignores_extra_whitespace():
    assert parse_moves("  R   U\tF2 \n") == ["R", "U", "F2"]


def test_parse_moves_empty():
    assert parse_moves("") == []


@pytest.mark.parametrize("text", ["R X", "r u", "R3", "RU"])
def test_parse_moves_rejects_unknown_moves(text):
    with pytest.raises(ValueError, match="Unknown move"):
        parse_moves(text)


def test_invert_moves():
    assert invert_moves(["R", "U", "R'", "U2"]) == ["U2", "R", "U'", "R'"]


def test_invert_moves_does_not_modify_input():
    moves = ["R", "U'"]
    invert_moves(moves)
    assert moves == ["R", "U'"]


def test_invert_moves_undoes_scramble():
    scramble = parse_moves("R U2 F' L D B2 R' U F2 D'")
    scrambled = _apply(SOLVED_STICKERS, scramble)
    assert np.array_equal(_apply(scrambled, invert_moves(scramble)), SOLVED_STICKERS)
