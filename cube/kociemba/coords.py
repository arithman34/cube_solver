from math import comb, factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cube.cubies import NUM_CORNERS, NUM_EDGES, CubieCube

NUM_UD_EDGES = 8
NUM_SLICE_EDGES = 4

N_TWIST = 3 ** (NUM_CORNERS - 1)
N_FLIP = 2 ** (NUM_EDGES - 1)
N_SLICE = comb(NUM_EDGES, NUM_SLICE_EDGES)
N_CORNERS = factorial(NUM_CORNERS)
N_UD_EDGES = factorial(NUM_UD_EDGES)
N_SLICE_PERM = factorial(NUM_SLICE_EDGES)

SLICE_EDGES = (8, 9, 10, 11)  # FR, FL, BL, BR


def _cube(
    corner_perm: ArrayLike | None = None,
    corner_orient: ArrayLike | None = None,
    edge_perm: ArrayLike | None = None,
    edge_orient: ArrayLike | None = None,
) -> CubieCube:
    """A CubieCube where any part not given is solved."""
    solved = CubieCube.solved()
    return CubieCube(
        corner_perm=solved.corner_perm
        if corner_perm is None
        else np.asarray(corner_perm),
        corner_orient=solved.corner_orient
        if corner_orient is None
        else np.asarray(corner_orient),
        edge_perm=solved.edge_perm if edge_perm is None else np.asarray(edge_perm),
        edge_orient=solved.edge_orient
        if edge_orient is None
        else np.asarray(edge_orient),
    )


def _rank_permutation(perm: ArrayLike) -> int:
    """Position of perm among all orderings of 0..n-1, from 0 to n! - 1. Identity is 0."""
    perm = np.asarray(perm)
    rank = 0
    for i in range(len(perm)):
        larger_before = int(np.sum(perm[:i] > perm[i]))
        rank = rank * (i + 1) + larger_before
    return rank


def _unrank_permutation(rank: int, size: int) -> NDArray[np.intp]:
    """The permutation of 0..size-1 with this rank. Inverse of _rank_permutation."""
    digits = [0] * size
    for i in range(size - 1, -1, -1):
        digits[i] = rank % (i + 1)
        rank //= i + 1

    unused = list(range(size))
    perm = np.empty(size, dtype=np.intp)
    for i in range(size - 1, -1, -1):
        perm[i] = unused.pop(len(unused) - 1 - digits[i])
    return perm


def get_twist(cubies: CubieCube) -> int:
    """Corner orientations as one number, 0 to 2186. Solved is 0."""
    twist = 0
    for i in range(NUM_CORNERS - 1):
        twist += int(cubies.corner_orient[i]) * 3**i
    return twist


def set_twist(twist: int) -> CubieCube:
    """A cube with this twist; every other part solved."""
    corner_orient = np.zeros(NUM_CORNERS, dtype=np.intp)
    for i in range(NUM_CORNERS - 1):
        corner_orient[i] = twist % 3
        twist //= 3
    corner_orient[-1] = -corner_orient[:-1].sum() % 3
    return _cube(corner_orient=corner_orient)


def get_flip(cubies: CubieCube) -> int:
    """Edge orientations as one number, 0 to 2047. Solved is 0."""
    flip = 0
    for i in range(NUM_EDGES - 1):
        flip += int(cubies.edge_orient[i]) * 2**i
    return flip


def set_flip(flip: int) -> CubieCube:
    """A cube with this flip; every other part solved."""
    edge_orient = np.zeros(NUM_EDGES, dtype=np.intp)
    for i in range(NUM_EDGES - 1):
        edge_orient[i] = flip % 2
        flip //= 2
    edge_orient[-1] = edge_orient[:-1].sum() % 2
    return _cube(edge_orient=edge_orient)


def get_slice(cubies: CubieCube) -> int:
    """Which 4 of the 12 edge slots hold a middle-layer edge, 0 to 494. Solved is 0."""
    slots = [slot for slot in range(NUM_EDGES) if cubies.edge_perm[slot] in SLICE_EDGES]
    from_end = sorted(NUM_EDGES - 1 - slot for slot in slots)
    return sum(comb(c, k + 1) for k, c in enumerate(from_end))


def set_slice(slice_: int) -> CubieCube:
    """A cube with the middle-layer edges in those slots; other edges fill the rest."""
    from_end = [0] * NUM_SLICE_EDGES
    for k in range(NUM_SLICE_EDGES - 1, -1, -1):
        c = k
        while comb(c + 1, k + 1) <= slice_:
            c += 1
        from_end[k] = c
        slice_ -= comb(c, k + 1)
    slots = sorted(NUM_EDGES - 1 - c for c in from_end)

    edge_perm = np.empty(NUM_EDGES, dtype=np.intp)
    slice_edges = iter(SLICE_EDGES)
    other_edges = iter(range(NUM_UD_EDGES))
    for slot in range(NUM_EDGES):
        edge_perm[slot] = next(slice_edges) if slot in slots else next(other_edges)
    return _cube(edge_perm=edge_perm)


def get_corners(cubies: CubieCube) -> int:
    """Corner permutation as one number, 0 to 40319. Solved is 0."""
    return _rank_permutation(cubies.corner_perm)


def set_corners(corners: int) -> CubieCube:
    """A cube with this corner permutation; every other part solved."""
    return _cube(corner_perm=_unrank_permutation(corners, NUM_CORNERS))


def get_ud_edges(cubies: CubieCube) -> int:
    """Permutation of the 8 top/bottom edges, 0 to 40319. Solved is 0."""
    return _rank_permutation(cubies.edge_perm[:NUM_UD_EDGES])


def set_ud_edges(ud_edges: int) -> CubieCube:
    """A cube with this top/bottom edge permutation; every other part solved."""
    edge_perm = np.concatenate(
        [_unrank_permutation(ud_edges, NUM_UD_EDGES), np.array(SLICE_EDGES)]
    )
    return _cube(edge_perm=edge_perm)


def get_slice_perm(cubies: CubieCube) -> int:
    """Order of the 4 middle-layer edges, 0 to 23. Solved is 0."""
    return _rank_permutation(cubies.edge_perm[NUM_UD_EDGES:] - NUM_UD_EDGES)


def set_slice_perm(slice_perm: int) -> CubieCube:
    """A cube with the middle-layer edges in this order; every other part solved."""
    edge_perm = np.concatenate(
        [
            np.arange(NUM_UD_EDGES),
            _unrank_permutation(slice_perm, NUM_SLICE_EDGES) + NUM_UD_EDGES,
        ]
    )
    return _cube(edge_perm=edge_perm)
