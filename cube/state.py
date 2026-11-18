from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cube.faces import (
    FACE_SIZE,
    NUM_FACES,
    NUM_STICKERS_PER_FACE,
    TOTAL_STICKERS,
    Face,
    face_from_letter,
)

SOLVED_STICKERS: NDArray[np.uint8] = np.repeat(
    np.arange(NUM_FACES, dtype=np.uint8), NUM_STICKERS_PER_FACE
)
SOLVED_STICKERS.flags.writeable = False

LETTERS = np.array([face.name for face in Face])


class Cube:
    """
    One state of a 3x3 cube, stored as 54 face values (0-5).

    Cubes are immutable, every operation returns a new Cube.
    """

    __array_ufunc__ = None

    def __init__(self, stickers: ArrayLike) -> None:
        """Validate shape and value range, then store a read-only uint8 copy."""
        state = np.array(stickers)
        if state.shape != (TOTAL_STICKERS,):
            raise ValueError(
                f"Shape of cube must be ({TOTAL_STICKERS},). Got {state.shape!r}."
            )

        if not np.issubdtype(state.dtype, np.integer):
            raise ValueError(f"Values must be integers. Got {state.dtype}")

        if state.min() < 0 or state.max() >= NUM_FACES:
            raise ValueError(
                f"Values must be 0-{NUM_FACES - 1}. Got min {state.min()!r}, max {state.max()!r}."
            )

        state = state.astype(np.uint8)
        state.flags.writeable = False
        self._stickers = state

    @classmethod
    def solved(cls) -> Self:
        """Return a solved cube."""
        return cls(SOLVED_STICKERS)

    @property
    def stickers(self) -> NDArray[np.uint8]:
        """The 54 sticker values (read-only)."""
        return self._stickers

    def is_solved(self) -> bool:
        """True if every sticker matches its face."""
        return np.array_equal(self.stickers, SOLVED_STICKERS)

    def __str__(self) -> str:
        """The unfolded net with face letters."""
        sep = " | "
        grid = LETTERS[self.stickers].reshape(NUM_FACES, FACE_SIZE, FACE_SIZE)

        def row_text(face: Face, row: int) -> str:
            return " ".join(grid[face][row])

        indent = " " * (len(row_text(Face.U, 0)) + len(sep))
        middle_faces = (Face.L, Face.F, Face.R, Face.B)

        lines = []
        for row in range(FACE_SIZE):
            lines.append(indent + row_text(Face.U, row))

        for row in range(FACE_SIZE):
            lines.append(sep.join(row_text(face, row) for face in middle_faces))

        for row in range(FACE_SIZE):
            lines.append(indent + row_text(Face.D, row))

        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        """Equal if the other object is a Cube with identical stickers."""
        if not isinstance(other, Cube):
            return NotImplemented

        return np.array_equal(self.stickers, other.stickers)

    def __hash__(self) -> int:
        """Hash of raw sticker bytes."""
        return hash(self._stickers.tobytes())

    def __repr__(self) -> str:
        """One-line form."""
        return f"{type(self).__name__}.from_string({self.to_string()!r})"

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Build a cube from 54 face letters, e.g. 'UUUUUUUUURRR...'."""
        if len(s) != TOTAL_STICKERS:
            raise ValueError(f"Expected {TOTAL_STICKERS} letters. Got {len(s)}.")

        stickers = [face_from_letter(char) for char in s]
        return cls(stickers)

    def to_string(self) -> str:
        """Return the 54-letter form."""
        return "".join(LETTERS[self.stickers])

    def to_array(self) -> NDArray[np.uint8]:
        """Return a writable copy of the stickers."""
        return self.stickers.copy()

    def apply_move(self, move: str) -> Self:
        """Return a new cube with one move applied."""
        raise NotImplementedError()

    def apply_moves(self, moves: str | list[str]) -> Self:
        """Return a new cube with a sequence applied."""
        if isinstance(moves, str):
            moves = moves.split()

        current = self
        for move in moves:
            current = current.apply_move(move)

        return current

    @classmethod
    def scramble(cls, num_moves: int, seed: int | None = None) -> Self:
        """Return a cube scramble by random moves (repeatable with a seed)."""
        raise NotImplementedError()
