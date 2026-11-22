from collections.abc import Iterator

from cube.cubies import CubieCube
from cube.solver.coords import (
    get_corners,
    get_flip,
    get_slice,
    get_slice_perm,
    get_twist,
    get_ud_edges,
)
from cube.solver.tables import PHASE_1_MOVES, PHASE_2_MOVES, Tables, get_tables

MAX_PHASE_1_DEPTH = 12
MAX_PHASE_2_DEPTH = 18

_BANNED_AFTER = {"D": "U", "L": "R", "B": "F"}  # U D and D U are the same cube


def _allowed(previous: str | None, move: str) -> bool:
    """Whether move may follow previous. False for the same face twice,
    or an opposite pair in the wrong order."""
    if previous is None:
        return True
    previous_face, face = previous[0], move[0]
    if face == previous_face:
        return False
    return _BANNED_AFTER.get(previous_face) != face


def phase_1_heuristic(tables: Tables, twist: int, flip: int, slice_: int) -> int:
    """Lower bound on the moves left to finish phase 1."""
    return max(
        int(tables.twist_slice_prune[twist, slice_]),
        int(tables.flip_slice_prune[flip, slice_]),
    )


def phase_2_heuristic(
    tables: Tables, corners: int, ud_edges: int, slice_perm: int
) -> int:
    """Lower bound on the moves left to finish phase 2."""
    return max(
        int(tables.corners_slice_perm_prune[corners, slice_perm]),
        int(tables.ud_edges_slice_perm_prune[ud_edges, slice_perm]),
    )


def _phase_1_solutions(
    tables: Tables, twist: int, flip: int, slice_: int, depth_left: int, path: list[int]
) -> Iterator[list[int]]:
    """Yield every phase-1 solution of exactly depth_left more moves, as indices
    into PHASE_1_MOVES (including the moves already in path). Solutions ending in
    a phase-2 move are skipped, because a shorter solution already covers them.
    path is left as it was when the search finishes."""
    if depth_left == 0:
        in_phase_2 = twist == 0 and flip == 0 and slice_ == 0
        if in_phase_2 and (not path or PHASE_1_MOVES[path[-1]] not in PHASE_2_MOVES):
            yield list(path)
        return
    if phase_1_heuristic(tables, twist, flip, slice_) > depth_left:
        return

    previous = PHASE_1_MOVES[path[-1]] if path else None
    for m, move in enumerate(PHASE_1_MOVES):
        if not _allowed(previous, move):
            continue
        path.append(m)
        yield from _phase_1_solutions(
            tables,
            int(tables.twist_move[twist, m]),
            int(tables.flip_move[flip, m]),
            int(tables.slice_move[slice_, m]),
            depth_left - 1,
            path,
        )
        path.pop()


def _phase_2_dfs(
    tables: Tables,
    corners: int,
    ud_edges: int,
    slice_perm: int,
    depth_left: int,
    path: list[int],
    previous: str | None,
) -> bool:
    """Depth-first search for a phase-2 finish using exactly depth_left more moves.
    path holds the indices into PHASE_2_MOVES chosen so far; on success it holds
    the whole solution and this returns True. On failure path is left as it was.
    previous is the move made just before path started, if any."""
    if depth_left == 0:
        return corners == 0 and ud_edges == 0 and slice_perm == 0
    if phase_2_heuristic(tables, corners, ud_edges, slice_perm) > depth_left:
        return False

    last = PHASE_2_MOVES[path[-1]] if path else previous
    for m, move in enumerate(PHASE_2_MOVES):
        if not _allowed(last, move):
            continue
        path.append(m)
        if _phase_2_dfs(
            tables,
            int(tables.corners_move[corners, m]),
            int(tables.ud_edges_move[ud_edges, m]),
            int(tables.slice_perm_move[slice_perm, m]),
            depth_left - 1,
            path,
            previous,
        ):
            return True
        path.pop()
    return False


def phase_1_solutions(
    cubies: CubieCube, max_depth: int = MAX_PHASE_1_DEPTH
) -> Iterator[list[str]]:
    """Yield the moves of every phase-1 solution up to max_depth moves, shortest
    first. Solutions that end in a phase-2 move are left out."""
    tables = get_tables()
    twist, flip, slice_ = get_twist(cubies), get_flip(cubies), get_slice(cubies)

    for depth in range(max_depth + 1):
        for path in _phase_1_solutions(tables, twist, flip, slice_, depth, []):
            yield [PHASE_1_MOVES[m] for m in path]


def search_phase_1(cubies: CubieCube, max_depth: int = MAX_PHASE_1_DEPTH) -> list[str]:
    """Shortest moves that bring twist, flip and slice to 0.
    Raises ValueError if none is found within max_depth."""
    for solution in phase_1_solutions(cubies, max_depth):
        return solution
    raise ValueError(f"No phase-1 solution within {max_depth} moves.")


def search_phase_2(
    cubies: CubieCube,
    max_depth: int = MAX_PHASE_2_DEPTH,
    previous: str | None = None,
) -> list[str]:
    """Shortest phase-2 moves that solve a cube already in phase 2. If previous is
    given, the solution does not start with a move on that face or one that may not
    follow it. Raises ValueError if the cube is not in phase 2 (twist, flip, slice
    not all 0) or no solution is found within max_depth."""
    if (get_twist(cubies), get_flip(cubies), get_slice(cubies)) != (0, 0, 0):
        raise ValueError("Cube is not in phase 2.")

    tables = get_tables()
    corners = get_corners(cubies)
    ud_edges = get_ud_edges(cubies)
    slice_perm = get_slice_perm(cubies)

    for depth in range(max_depth + 1):
        path: list[int] = []
        if _phase_2_dfs(tables, corners, ud_edges, slice_perm, depth, path, previous):
            return [PHASE_2_MOVES[m] for m in path]
    raise ValueError(f"No phase-2 solution within {max_depth} moves.")
