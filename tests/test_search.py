from itertools import islice, pairwise

import pytest

from cube.cubies import CubieCube
from cube.moves import MOVE_NAMES
from cube.solver import solve
from cube.solver.coords import get_flip, get_slice, get_twist
from cube.solver.search import (
    MAX_PHASE_1_DEPTH,
    PHASE_2_MOVES,
    _allowed,
    phase_1_heuristic,
    phase_1_solutions,
    phase_2_heuristic,
    search_phase_1,
    search_phase_2,
)
from cube.solver.tables import get_tables
from cube.state import Cube


def _cubies(moves: str) -> CubieCube:
    cubies = CubieCube.solved()
    for move in moves.split():
        cubies = cubies.apply_move(move)
    return cubies


def _no_wasted_moves(moves: list[str]) -> bool:
    return all(_allowed(a, b) for a, b in pairwise(moves))


@pytest.mark.parametrize(
    "previous, move, expected",
    [
        (None, "R", True),
        ("U", "D", True),
        ("U'", "R2", True),
        ("R", "R'", False),
        ("D2", "D", False),
        ("D'", "U2", False),
        ("L", "R", False),
        ("B", "F'", False),
        ("R", "L", True),
    ],
)
def test_allowed(previous, move, expected):
    assert _allowed(previous, move) is expected


def test_heuristics_are_zero_when_solved():
    tables = get_tables()
    assert phase_1_heuristic(tables, 0, 0, 0) == 0
    assert phase_2_heuristic(tables, 0, 0, 0) == 0


def test_phase_1_heuristic_after_r():
    assert phase_1_heuristic(get_tables(), 110, 0, 367) == 1


def test_phase_1_on_solved_cube_is_empty():
    assert search_phase_1(CubieCube.solved()) == []


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("R", ["R"]),
        ("F", ["F"]),
        ("R U", ["U'", "R"]),
        ("R U F", ["F'", "U'", "R"]),
        ("F R B L", ["L'", "B'", "R'", "F"]),
    ],
)
def test_phase_1_short_scrambles(moves, expected):
    assert search_phase_1(_cubies(moves)) == expected


def test_phase_1_is_empty_for_a_cube_already_in_phase_2():
    assert search_phase_1(_cubies("R2 U F2 D'")) == []


@pytest.mark.parametrize("seed", range(5))
def test_phase_1_reaches_phase_2(seed):
    cubies = CubieCube.from_stickers(Cube.scramble(30, seed=seed).stickers)
    moves = search_phase_1(cubies)
    for move in moves:
        cubies = cubies.apply_move(move)
    assert (get_twist(cubies), get_flip(cubies), get_slice(cubies)) == (0, 0, 0)
    assert len(moves) <= MAX_PHASE_1_DEPTH
    assert _no_wasted_moves(moves)


def test_phase_1_gives_up_past_max_depth():
    with pytest.raises(ValueError):
        search_phase_1(_cubies("R U F"), max_depth=2)


def test_phase_1_solutions_come_shortest_first():
    cubies = _cubies("R U F")
    solutions = list(islice(phase_1_solutions(cubies), 40))

    assert solutions[0] == search_phase_1(cubies)
    assert [len(s) for s in solutions] == sorted(len(s) for s in solutions)
    for moves in solutions:
        assert moves[-1] not in PHASE_2_MOVES
        reached = cubies
        for move in moves:
            reached = reached.apply_move(move)
        assert (get_twist(reached), get_flip(reached), get_slice(reached)) == (0, 0, 0)


def test_phase_2_on_solved_cube_is_empty():
    assert search_phase_2(CubieCube.solved()) == []


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("U", ["U'"]),
        ("R2 U", ["U'", "R2"]),
        ("U R2 D F2", ["F2", "D'", "R2", "U'"]),
        ("R2 U2 L2 D B2", ["B2", "D'", "L2", "U2", "R2"]),
    ],
)
def test_phase_2_short_scrambles(moves, expected):
    assert search_phase_2(_cubies(moves)) == expected


def test_phase_2_only_uses_phase_2_moves():
    moves = search_phase_2(_cubies("R2 U F2 D' L2 B2 U2 D R2 U'"))
    assert set(moves) <= set(PHASE_2_MOVES)
    assert _no_wasted_moves(moves)


def test_phase_2_does_not_start_on_the_previous_face():
    assert search_phase_2(_cubies("U R2 D F2"), previous="F") == [
        "L2",
        "U'",
        "D",
        "F2",
        "R2",
        "D'",
        "B2",
        "D'",
    ]


def test_phase_2_gives_up_past_max_depth():
    with pytest.raises(ValueError):
        search_phase_2(_cubies("R2 U"), max_depth=1)


def test_phase_2_rejects_cube_not_in_phase_2():
    with pytest.raises(ValueError, match="phase 2"):
        search_phase_2(_cubies("R"))


def test_solve_solved_cube():
    assert solve(Cube.solved()) == []


@pytest.mark.parametrize(
    "moves, expected",
    [
        ("R", ["R'"]),
        ("L", ["L'"]),
        ("F'", ["F"]),
        ("B2", ["B2"]),
        ("U", ["U'"]),
        ("R U", ["U'", "R'"]),
        ("R U R' U'", ["U", "R", "U'", "R'"]),
    ],
)
def test_solve_short_scrambles_exactly(moves, expected):
    assert solve(Cube.solved().apply_moves(moves)) == expected


@pytest.mark.parametrize("seed", range(10))
def test_solve_scrambled_cube(seed):
    cube = Cube.scramble(30, seed=seed)
    moves = solve(cube, time_limit=0.2)
    assert cube.apply_moves(moves).is_solved()
    assert set(moves) <= set(MOVE_NAMES)
    assert all(a[0] != b[0] for a, b in pairwise(moves))
    assert len(moves) <= 22


def test_first_solution_is_already_short():
    lengths = [
        len(solve(Cube.scramble(30, seed=seed), time_limit=0)) for seed in range(5)
    ]
    assert lengths == [22, 21, 20, 21, 21]


def test_more_time_never_gives_a_longer_solution():
    cube = Cube.scramble(30, seed=4)
    quick = solve(cube, time_limit=0)
    longer = solve(cube, time_limit=0.5)
    assert cube.apply_moves(longer).is_solved()
    assert len(longer) <= len(quick)


def test_solve_rejects_unsolvable_cube():
    stickers = Cube.solved().to_array()
    stickers[[5, 10]] = stickers[[10, 5]]
    with pytest.raises(ValueError):
        solve(Cube(stickers))
