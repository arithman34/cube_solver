from cube.cfop.algorithms import simplify
from cube.cfop.cross import solve_cross
from cube.cfop.f2l import solve_f2l
from cube.cfop.last_layer import solve_oll, solve_pll
from cube.cubies import CubieCube
from cube.state import Cube
from cube.validation import is_solvable


def _after(cubies: CubieCube, moves: list[str]) -> CubieCube:
    """The cube after moves."""
    for move in moves:
        cubies = cubies.apply_move(move)

    return cubies


def solve(cube: Cube, time_limit: float = 0.0) -> list[str]:
    """Moves that solve the cube the CFOP way."""
    if not is_solvable(cube):
        raise ValueError("Cube is not solvable.")

    cubies = CubieCube.from_stickers(cube.stickers)
    cross = solve_cross(cubies)
    cubies = _after(cubies, cross)

    f2l = solve_f2l(cubies, cross[-1] if cross else None)
    cubies = _after(cubies, f2l)

    oll = solve_oll(cubies)
    cubies = _after(cubies, oll)

    pll = solve_pll(cubies)

    return simplify(cross + f2l + oll + pll)
