from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cube.faces import NUM_STICKERS_PER_FACE, Face, index
from cube.moves import MOVE_NAMES, apply_move
from cube.state import SOLVED_STICKERS

U, R, F, D, L, B = Face

NUM_CORNERS = 8
NUM_EDGES = 12

CORNER_NAMES: tuple[str, ...] = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
EDGE_NAMES: tuple[str, ...] = (
    "UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR",
)  # fmt: skip

CORNER_FACELETS: NDArray[np.intp] = np.array(
    [
        [index(U, 2, 2), index(R, 0, 0), index(F, 0, 2)],  # URF
        [index(U, 2, 0), index(F, 0, 0), index(L, 0, 2)],  # UFL
        [index(U, 0, 0), index(L, 0, 0), index(B, 0, 2)],  # ULB
        [index(U, 0, 2), index(B, 0, 0), index(R, 0, 2)],  # UBR
        [index(D, 0, 2), index(F, 2, 2), index(R, 2, 0)],  # DFR
        [index(D, 0, 0), index(L, 2, 2), index(F, 2, 0)],  # DLF
        [index(D, 2, 0), index(B, 2, 2), index(L, 2, 0)],  # DBL
        [index(D, 2, 2), index(R, 2, 2), index(B, 2, 0)],  # DRB
    ]
)

EDGE_FACELETS: NDArray[np.intp] = np.array(
    [
        [index(U, 1, 2), index(R, 0, 1)],  # UR
        [index(U, 2, 1), index(F, 0, 1)],  # UF
        [index(U, 1, 0), index(L, 0, 1)],  # UL
        [index(U, 0, 1), index(B, 0, 1)],  # UB
        [index(D, 1, 2), index(R, 2, 1)],  # DR
        [index(D, 0, 1), index(F, 2, 1)],  # DF
        [index(D, 1, 0), index(L, 2, 1)],  # DL
        [index(D, 2, 1), index(B, 2, 1)],  # DB
        [index(F, 1, 2), index(R, 1, 0)],  # FR
        [index(F, 1, 0), index(L, 1, 2)],  # FL
        [index(B, 1, 2), index(L, 1, 0)],  # BL
        [index(B, 1, 0), index(R, 1, 2)],  # BR
    ]
)

CORNER_FACES: NDArray[np.intp] = CORNER_FACELETS // NUM_STICKERS_PER_FACE
EDGE_FACES: NDArray[np.intp] = EDGE_FACELETS // NUM_STICKERS_PER_FACE


@dataclass(frozen=True)
class CubieCube:
    """A cube described by its 8 corners and 12 edges."""

    corner_perm: NDArray[np.intp]
    corner_orient: NDArray[np.intp]
    edge_perm: NDArray[np.intp]
    edge_orient: NDArray[np.intp]

    __array_ufunc__ = None

    @classmethod
    def solved(cls) -> Self:
        """Every piece in its home slot, orientation 0."""
        return cls(
            corner_perm=np.arange(NUM_CORNERS),
            corner_orient=np.zeros(NUM_CORNERS, dtype=np.intp),
            edge_perm=np.arange(NUM_EDGES),
            edge_orient=np.zeros(NUM_EDGES, dtype=np.intp),
        )

    @classmethod
    def from_stickers(cls, stickers: ArrayLike) -> Self:
        """Read pieces from 54 sticker values; ValueError if a piece doesn't exist."""
        stickers = np.asarray(stickers)

        corner_perm = np.empty(NUM_CORNERS, dtype=np.intp)
        corner_orient = np.empty(NUM_CORNERS, dtype=np.intp)
        for slot in range(NUM_CORNERS):
            faces = tuple(int(face) for face in stickers[CORNER_FACELETS[slot]])
            corner_perm[slot], corner_orient[slot] = _find_corner(faces)

        edge_perm = np.empty(NUM_EDGES, dtype=np.intp)
        edge_orient = np.empty(NUM_EDGES, dtype=np.intp)
        for slot in range(NUM_EDGES):
            faces = tuple(int(face) for face in stickers[EDGE_FACELETS[slot]])
            edge_perm[slot], edge_orient[slot] = _find_edge(faces)

        if len(set(corner_perm)) != NUM_CORNERS:
            raise ValueError("The same corner appears more than once.")
        if len(set(edge_perm)) != NUM_EDGES:
            raise ValueError("The same edge appears more than once.")

        return cls(corner_perm, corner_orient, edge_perm, edge_orient)

    def multiply(self, other: "CubieCube") -> Self:
        """The cube you get by doing self, then other."""
        new_corner_perm = self.corner_perm[other.corner_perm]
        new_corner_orient = (self.corner_orient[other.corner_perm] + other.corner_orient) % 3

        new_edge_perm = self.edge_perm[other.edge_perm]
        new_edge_orient = (self.edge_orient[other.edge_perm] + other.edge_orient) % 2

        return type(self)(new_corner_perm, new_corner_orient, new_edge_perm, new_edge_orient)

    def apply_move(self, move: str) -> Self:
        """Return a new CubieCube with one move applied."""
        if move not in MOVE_CUBIES:
            raise ValueError(f"Unknown move. Got {move!r}.")

        return self.multiply(MOVE_CUBIES[move])

    def __eq__(self, other: object) -> bool:
        """Equal if all four arrays match."""
        if not isinstance(other, CubieCube):
            return NotImplemented

        return all([
            np.array_equal(self.corner_perm, other.corner_perm),
            np.array_equal(self.corner_orient, other.corner_orient),
            np.array_equal(self.edge_perm, other.edge_perm),
            np.array_equal(self.edge_orient, other.edge_orient)
        ])


def _build_move_cubies() -> dict[str, CubieCube]:
    """All 18 moves as CubieCubes, read from the sticker moves."""
    return {
        name: CubieCube.from_stickers(apply_move(SOLVED_STICKERS, name))
        for name in MOVE_NAMES
    }


def _find_corner(faces: tuple[int, ...]) -> tuple[int, int]:
    """(corner id, orientation) for the 3 faces read from a slot; ValueError if none match."""
    for corner_id, corner_faces in enumerate(CORNER_FACES):
        if set(faces) == set(corner_faces.tolist()):
            for orient, face in enumerate(faces):
                if face in (Face.U, Face.D):
                    return corner_id, orient
    raise ValueError(f"No corner has faces {faces}.")


def _find_edge(faces: tuple[int, ...]) -> tuple[int, int]:
    """(edge id, orientation) for the 2 faces read from a slot; ValueError if none match."""
    for edge_id, edge_faces in enumerate(EDGE_FACES):
        if set(faces) == set(edge_faces.tolist()):
            orient = 0 if faces[0] == edge_faces[0] else 1
            return edge_id, orient
    raise ValueError(f"No edge has faces {faces}.")


def permutation_parity(perm: ArrayLike) -> int:
    """0 if perm is an even permutation, 1 if odd."""
    perm = np.asarray(perm)
    seen = [False] * len(perm)
    cycles = 0
    for start in range(len(perm)):
        if seen[start]:
            continue
        cycles += 1
        i = start
        while not seen[i]:
            seen[i] = True
            i = perm[i]
    return (len(perm) - cycles) % 2


MOVE_CUBIES: dict[str, CubieCube] = _build_move_cubies()
