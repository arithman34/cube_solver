from enum import IntEnum


class Face(IntEnum):
    """The six faces in Kociemba order.

    A face's value times NUM_STICKERS_PER_FACE is the index of its first
    sticker (U -> 0, R -> 9, F -> 18, ...). Changing this order changes the
    whole 54-sticker layout, so it must stay fixed.
    """

    U = 0
    R = 1
    F = 2
    D = 3
    L = 4
    B = 5


NUM_FACES = len(Face)
FACE_SIZE = 3
MIDDLE = FACE_SIZE // 2
NUM_STICKERS_PER_FACE = FACE_SIZE * FACE_SIZE
TOTAL_STICKERS = NUM_FACES * NUM_STICKERS_PER_FACE

_OPPOSITE = {
    Face.U: Face.D,
    Face.D: Face.U,
    Face.R: Face.L,
    Face.L: Face.R,
    Face.F: Face.B,
    Face.B: Face.F,
}


def index(face: Face | int, row: int, col: int) -> int:
    """Return the flat 0-53 index of the sticker at (face, row, col)."""
    face = Face(face)  # raises ValueError for anything that isn't 0-5

    if row < 0 or row >= FACE_SIZE:
        raise ValueError(f"Row must be 0 <= row < {FACE_SIZE}. Got {row}.")

    if col < 0 or col >= FACE_SIZE:
        raise ValueError(f"Column must be 0 <= column < {FACE_SIZE}. Got {col}.")

    return face * NUM_STICKERS_PER_FACE + row * FACE_SIZE + col


def center_index(face: Face | int) -> int:
    """Return the flat index of a face's center sticker."""
    return index(face, MIDDLE, MIDDLE)


def opposite(face: Face | int) -> Face:
    """Return the face across the cube from the given face."""
    return _OPPOSITE[Face(face)]


def face_from_letter(letter: str) -> Face:
    """Convert a letter such as 'U' to Face.U."""
    try:
        return Face[letter]
    except KeyError:
        raise ValueError(
            f"Unknown face letter {letter!r}. Expected one of "
            f"{', '.join(f.name for f in Face)}."
        ) from None
