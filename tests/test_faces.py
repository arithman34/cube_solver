import pytest

from cube.faces import (
    Face,
    index,
    center_index,
    opposite,
    face_from_letter,
    NUM_FACES,
    FACE_SIZE,
    NUM_STICKERS_PER_FACE,
    TOTAL_STICKERS,
)


# index
@pytest.mark.parametrize(
    "face, row, col, expected",
    [
        (Face.U, 1, 2, 5),
        (Face.R, 2, 2, 17),
        (Face.F, 0, 0, 18),
        (Face.D, 1, 0, 30),
        (Face.L, 0, 2, 38),
        (Face.B, 2, 2, 53),
    ],
)
def test_index_returns_flat_position(face, row, col, expected):
    assert index(face, row, col) == expected


@pytest.mark.parametrize("face", list(Face))
def test_index_first_sticker_of_each_face(face):
    assert index(face, 0, 0) == face * NUM_STICKERS_PER_FACE


def test_index_covers_all_positions_exactly_once():
    unique = set()
    for face in Face:
        for row in range(FACE_SIZE):
            for col in range(FACE_SIZE):
                unique.add(index(face, row, col))

    all_positions = set(range(TOTAL_STICKERS))

    assert unique == all_positions


@pytest.mark.parametrize("row", [-1, FACE_SIZE, 10])
def test_index_rejects_row_out_of_range(row):
    with pytest.raises(ValueError):
        index(Face.U, row, 0)


@pytest.mark.parametrize("col", [-1, FACE_SIZE, 10])
def test_index_rejects_col_out_of_range(col):
    with pytest.raises(ValueError):
        index(Face.U, 0, col)


@pytest.mark.parametrize("face", [-1, 6, 10])
def test_index_rejects_invalid_face(face):
    with pytest.raises(ValueError):
        index(face, 0, 0)


def test_index_accepts_plain_int_face():
    assert index(3, 0, 0) == index(Face.D, 0, 0)


# center_index
@pytest.mark.parametrize(
    "face, expected",
    [(Face.U, 4), (Face.R, 13), (Face.F, 22), (Face.D, 31), (Face.L, 40), (Face.B, 49)],
)
def test_center_index_matches_known_centers(face, expected):
    assert center_index(face) == expected


@pytest.mark.parametrize("face", [-1, 6, 10])
def test_center_index_rejects_invalid_face(face):
    with pytest.raises(ValueError):
        center_index(face)


# opposite
@pytest.mark.parametrize(
    "face, expected",
    [
        (Face.U, Face.D),
        (Face.R, Face.L),
        (Face.F, Face.B),
        (Face.B, Face.F),
        (Face.L, Face.R),
        (Face.D, Face.U),
    ],
)
def test_opposite_known_pairs(face, expected):
    assert opposite(face) == expected


@pytest.mark.parametrize("face", list(Face))
def test_opposite_is_its_own_inverse(face):
    assert opposite(opposite(face)) == face


@pytest.mark.parametrize("face", list(Face))
def test_opposite_is_never_itself(face):
    assert opposite(face) != face


@pytest.mark.parametrize("face", [-1, 6, 10])
def test_opposite_rejects_invalid_face(face):
    with pytest.raises(ValueError):
        opposite(face)


# face_from_letter
@pytest.mark.parametrize(
    "letter, expected",
    [
        ("U", Face.U),
        ("R", Face.R),
        ("F", Face.F),
        ("D", Face.D),
        ("L", Face.L),
        ("B", Face.B),
    ],
)
def test_face_from_letter_valid(letter, expected):
    assert face_from_letter(letter) == expected


@pytest.mark.parametrize("face", list(Face))
def test_face_from_letter_round_trips_with_name(face):
    assert face_from_letter(face.name) == face


@pytest.mark.parametrize("letter", ["", "X", "UR", "A", "FLB"])
def test_face_from_letter_rejects_invalid(letter):
    with pytest.raises(ValueError):
        face_from_letter(letter)


@pytest.mark.parametrize(
    "letter",
    [
        "u",
        "r",
        "f",
    ],
)
def test_face_from_letter_rejects_lowercase(letter):
    with pytest.raises(ValueError):
        face_from_letter(letter)


# constants
def test_constants():
    assert NUM_FACES == 6
    assert NUM_STICKERS_PER_FACE == 9
    assert TOTAL_STICKERS == 54
    assert FACE_SIZE == 3
