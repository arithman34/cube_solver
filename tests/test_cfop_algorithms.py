import pytest

from cube.cfop.algorithms import OLL, PLL, expand, simplify
from cube.cfop.stages import is_f2l_solved, is_oll_solved
from cube.cubies import CubieCube
from cube.state import Cube


def _after(moves: list[str]) -> CubieCube:
    return CubieCube.from_stickers(Cube.solved().apply_moves(moves).stickers)


@pytest.mark.parametrize(
    "alg, expected",
    [
        ("R U R' U'", ["R", "U", "R'", "U'"]),
        ("y R U R'", ["B", "U", "B'"]),
        ("r U R' U' r' F R F'", ["L", "F", "R'", "F'", "L'", "F", "R", "F'"]),
        (
            "M2 U M2 U2 M2 U M2",
            ["R2", "L2", "D", "R2", "L2", "U2", "R2", "L2", "D", "R2", "L2"],
        ),
        (
            "x R' U R' D2 R U' R' D2 R2 x'",
            ["R'", "F", "R'", "B2", "R", "F'", "R'", "B2", "R2"],
        ),
        (
            "F R U R' U' F' f R U R' U' f'",
            ["F", "R", "U", "R'", "U'", "F'", "B", "U", "L", "U'", "L'", "B'"],
        ),
    ],
)
def test_expand(alg, expected):
    assert expand(alg) == expected


@pytest.mark.parametrize(
    "moves, expected",
    [
        ([], []),
        (["R", "U"], ["R", "U"]),
        (["R", "R"], ["R2"]),
        (["U'", "U2"], ["U"]),
        (["R", "R", "R", "R"], []),
        (["U", "D", "U"], ["U2", "D"]),
        (["R", "L", "R'", "L'"], []),
        (["F", "U", "U'", "F"], ["F2"]),
    ],
)
def test_simplify(moves, expected):
    assert simplify(moves) == expected


def test_there_are_57_oll_and_21_pll_algorithms():
    assert len(OLL) == 57
    assert len(PLL) == 21


@pytest.mark.parametrize("name", OLL)
def test_oll_algorithm_keeps_f2l(name):
    assert is_f2l_solved(_after(expand(OLL[name])))


@pytest.mark.parametrize("name", PLL)
def test_pll_algorithm_keeps_f2l_and_oll(name):
    assert is_oll_solved(_after(expand(PLL[name])))


@pytest.mark.parametrize("name", PLL)
def test_pll_algorithm_changes_the_cube(name):
    assert _after(expand(PLL[name])) != CubieCube.solved()
