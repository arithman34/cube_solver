from functools import cache

from cube.cfop.algorithms import OLL, PLL, expand, simplify
from cube.cfop.stages import is_f2l_solved, is_oll_solved, is_pll_solved
from cube.cubies import CubieCube

AUF = ([], ["U"], ["U2"], ["U'"])


def solve_oll(cubies: CubieCube) -> list[str]:
    """Shortest U turn plus OLL algorithm that makes the U face one colour."""
    if not is_f2l_solved(cubies):
        raise ValueError("F2L is not solved.")

    best = None
    for alg in [[], *_expanded(tuple(OLL.values()))]:
        for before in AUF:
            moves = simplify(before + alg)
            if is_oll_solved(_after(cubies, moves)) and _shorter(moves, best):
                best = moves
    if best is None:
        raise ValueError("No OLL algorithm fits this cube.")
    return best


def solve_pll(cubies: CubieCube) -> list[str]:
    """Shortest U turn, PLL algorithm and U turn that solve the cube."""
    if not is_oll_solved(cubies):
        raise ValueError("OLL is not solved.")

    best = None
    for alg in [[], *_expanded(tuple(PLL.values()))]:
        for before in AUF:
            for after in AUF:
                moves = simplify(before + alg + after)
                if is_pll_solved(_after(cubies, moves)) and _shorter(moves, best):
                    best = moves
    if best is None:
        raise ValueError("No PLL algorithm fits this cube.")
    return best


@cache
def _expanded(algorithms: tuple[str, ...]) -> list[list[str]]:
    """Each algorithm as outer face turns."""
    return [expand(alg) for alg in algorithms]


def _after(cubies: CubieCube, moves: list[str]) -> CubieCube:
    """The cube after moves."""
    for move in moves:
        cubies = cubies.apply_move(move)
    return cubies


def _shorter(moves: list[str], best: list[str] | None) -> bool:
    """Whether moves beats the best so far."""
    return best is None or len(moves) < len(best)
