from cube.cubies import CubieCube
from cube.solver.search import search_phase_1, search_phase_2
from cube.state import Cube
from cube.validation import is_solvable

_QUARTER_TURNS = {"": 1, "2": 2, "'": 3}
_SUFFIXES = {1: "", 2: "2", 3: "'"}


def solve(cube: Cube) -> list[str]:
    """Moves that solve cube. Empty for a solved cube. Raises ValueError if the cube is not solvable."""
    if not is_solvable(cube):
        raise ValueError("Cube is not solvable.")

    cubies = CubieCube.from_stickers(cube.stickers)

    phase_1 = search_phase_1(cubies)

    for move in phase_1:
        cubies = cubies.apply_move(move)

    phase_2 = search_phase_2(cubies)

    while phase_1 and phase_2 and phase_1[-1][0] == phase_2[0][0]:
        last_move, first_move = phase_1.pop(), phase_2.pop(0)
        face = last_move[0]
        turns = (_QUARTER_TURNS[last_move[1:]] + _QUARTER_TURNS[first_move[1:]]) % 4
        if turns != 0:
            phase_1.append(face + _SUFFIXES[turns])
            break

    return phase_1 + phase_2
