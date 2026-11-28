import time

from cube.cubies import CubieCube
from cube.kociemba.search import phase_1_solutions, search_phase_2
from cube.state import Cube
from cube.validation import is_solvable

MAX_PHASE_2_BUDGET = 10
DEFAULT_TIME_LIMIT = 1.0


def solve(cube: Cube, time_limit: float = DEFAULT_TIME_LIMIT) -> list[str]:
    """Moves that solve cube, as short as can be found within time_limit seconds.
    Always returns a solution, even if finding the first one takes longer.
    Empty for a solved cube. Raises ValueError if the cube is not solvable."""
    if not is_solvable(cube):
        raise ValueError("Cube is not solvable.")

    deadline = time.perf_counter() + time_limit
    cubies = CubieCube.from_stickers(cube.stickers)
    best: list[str] | None = None

    for phase_1 in phase_1_solutions(cubies):
        if best is not None:
            if len(phase_1) >= len(best) or time.perf_counter() > deadline:
                break
            budget = min(MAX_PHASE_2_BUDGET, len(best) - len(phase_1) - 1)
        else:
            budget = MAX_PHASE_2_BUDGET

        in_phase_2 = cubies
        for move in phase_1:
            in_phase_2 = in_phase_2.apply_move(move)

        previous = phase_1[-1] if phase_1 else None
        try:
            phase_2 = search_phase_2(in_phase_2, max_depth=budget, previous=previous)
        except ValueError:
            continue
        best = phase_1 + phase_2

    if best is None:
        raise ValueError("No solution found.")
    return best
