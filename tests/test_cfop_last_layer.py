from itertools import permutations, product

import numpy as np
import pytest

from cube.cfop.algorithms import PLL, expand
from cube.cfop.cross import solve_cross
from cube.cfop.f2l import solve_f2l
from cube.cfop.last_layer import solve_oll, solve_pll
from cube.cfop.stages import is_oll_solved, is_pll_solved
from cube.cubies import CubieCube, permutation_parity
from cube.state import Cube

SUNE = "R U R' U R U2 R'"
T_PERM = "R U R' U' R' F R2 U' R' U' R U R' F'"


def _cubies(moves: str) -> CubieCube:
    return CubieCube.from_stickers(Cube.solved().apply_moves(moves).stickers)


def _after(cubies: CubieCube, moves: list[str]) -> CubieCube:
    for move in moves:
        cubies = cubies.apply_move(move)
    return cubies


def _last_layer(corner_perm, corner_orient, edge_perm, edge_orient) -> CubieCube:
    return CubieCube(
        np.array([*corner_perm, 4, 5, 6, 7]),
        np.array([*corner_orient, 0, 0, 0, 0]),
        np.array([*edge_perm, *range(4, 12)]),
        np.array([*edge_orient, *[0] * 8]),
    )


def _every_orientation():
    for corners in product(range(3), repeat=4):
        for edges in product(range(2), repeat=4):
            if sum(corners) % 3 == 0 and sum(edges) % 2 == 0:
                yield _last_layer(range(4), corners, range(4), edges)


def _every_permutation():
    for corners in permutations(range(4)):
        for edges in permutations(range(4)):
            if permutation_parity(corners) == permutation_parity(edges):
                yield _last_layer(corners, [0] * 4, edges, [0] * 4)


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", []),
        ("U", []),
        (T_PERM, []),
        ("F R U R' U' F'", ["F", "U", "R", "U'", "R'", "F'"]),
        (SUNE, ["R", "U2", "R'", "U'", "R", "U'", "R'"]),
        ("R U2 R' U' R U' R'", ["R", "U", "R'", "U", "R", "U2", "R'"]),
    ],
)
def test_solve_oll(moves, expected):
    assert solve_oll(_cubies(moves)) == expected


def test_solve_oll_handles_every_case():
    cases = list(_every_orientation())
    assert len(cases) == 216
    for cubies in cases:
        assert is_oll_solved(_after(cubies, solve_oll(cubies)))


def test_solve_oll_rejects_unsolved_f2l():
    with pytest.raises(ValueError, match="F2L"):
        solve_oll(_cubies("R"))


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("", []),
        ("U", ["U'"]),
        ("U2", ["U2"]),
        (T_PERM, T_PERM.split()),
        (f"U {T_PERM}", [*T_PERM.split(), "U'"]),
        (
            " ".join(expand(PLL["H"])),
            ["R2", "L2", "D", "R2", "L2", "U2", "R2", "L2", "D", "R2", "L2"],
        ),
    ],
)
def test_solve_pll(moves, expected):
    assert solve_pll(_cubies(moves)) == expected


def test_solve_pll_handles_every_case():
    cases = list(_every_permutation())
    assert len(cases) == 288
    for cubies in cases:
        assert is_pll_solved(_after(cubies, solve_pll(cubies)))


def test_solve_pll_rejects_unsolved_oll():
    with pytest.raises(ValueError, match="OLL"):
        solve_pll(_cubies(SUNE))


@pytest.mark.parametrize("seed", range(3))
def test_cfop_solves_a_scrambled_cube(seed):
    cubies = CubieCube.from_stickers(Cube.scramble(30, seed=seed).stickers)
    cross = solve_cross(cubies)
    cubies = _after(cubies, cross)
    f2l = solve_f2l(cubies, previous=cross[-1])
    cubies = _after(cubies, f2l)
    cubies = _after(cubies, solve_oll(cubies))
    cubies = _after(cubies, solve_pll(cubies))
    assert cubies == CubieCube.solved()
