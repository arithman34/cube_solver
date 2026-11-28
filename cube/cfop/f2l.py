from collections import deque
from functools import cache

from cube.cfop.cross import _cross_key, _distances, _edge_moves, _move_key
from cube.cfop.stages import solved_pairs
from cube.cubies import MOVE_CUBIES, NUM_CORNERS, CubieCube
from cube.solver.search import _allowed

MAX_PAIR_DEPTH = 20


def solve_f2l(cubies: CubieCube, previous: str | None = None) -> list[str]:
    """Moves that solve all four pairs, quickest pair first."""
    done = solved_pairs(cubies)
    moves = []

    while len(done) < 4:
        best_pair, best_moves = None, None
        for pair in range(4):
            if pair in done:
                continue
            pair_moves = solve_pair(cubies, pair, done, previous)
            if best_moves is None or len(pair_moves) < len(best_moves):
                best_pair, best_moves = pair, pair_moves

        for move in best_moves:
            cubies = cubies.apply_move(move)
        moves += best_moves
        done.append(best_pair)
        if moves:
            previous = moves[-1]

    return moves


def solve_pair(
    cubies: CubieCube, pair: int, done: list[int], previous: str | None = None
) -> list[str]:
    """Shortest moves that solve pair without breaking the cross or done pairs."""
    cross_key = _cross_key(cubies)
    pair_keys = {p: _pair_key(cubies, p) for p in [*done, pair]}

    for depth in range(MAX_PAIR_DEPTH + 1):
        path: list[str] = []
        if _search(cross_key, pair_keys, depth, path, previous):
            return path
    raise ValueError(f"No way to solve pair {pair} within {MAX_PAIR_DEPTH} moves.")


def _search(
    cross_key: tuple[int, ...],
    pair_keys: dict[int, tuple[int, int]],
    depth_left: int,
    path: list[str],
    previous: str | None,
) -> bool:
    """Whether the pairs can be solved within depth_left moves; fills path if so."""
    bound = max(
        _distances()[cross_key],
        *(_pair_distances(p)[key] for p, key in pair_keys.items()),
    )
    if bound == 0:
        return True
    if bound > depth_left:
        return False

    last = path[-1] if path else previous
    for move, edge_table in _edge_moves().items():
        if not _allowed(last, move):
            continue
        corner_table = _corner_moves()[move]
        path.append(move)
        if _search(
            _move_key(cross_key, edge_table),
            {
                p: _move_pair_key(key, corner_table, edge_table)
                for p, key in pair_keys.items()
            },
            depth_left - 1,
            path,
            previous,
        ):
            return True
        path.pop()
    return False


def _pair_key(cubies: CubieCube, pair: int) -> tuple[int, int]:
    """Where a pair's corner and edge are."""
    corner_slot = list(cubies.corner_perm).index(4 + pair)
    twist = int(cubies.corner_orient[corner_slot])
    edge_slot = list(cubies.edge_perm).index(8 + pair)
    flip = int(cubies.edge_orient[edge_slot])
    return (3 * corner_slot + twist, 2 * edge_slot + flip)


def _move_pair_key(
    key: tuple[int, int], corner_table: list[int], edge_table: list[int]
) -> tuple[int, int]:
    """The pair key after a move."""
    corner, edge = key
    return (corner_table[corner], edge_table[edge])


@cache
def _corner_moves() -> dict[str, list[int]]:
    """Where each corner position goes after each move."""
    corner_moves = {}
    for move, cubies in MOVE_CUBIES.items():
        table = [0] * (3 * NUM_CORNERS)
        for new_slot in range(NUM_CORNERS):
            old_slot = int(cubies.corner_perm[new_slot])
            for twist in range(3):
                new_twist = (twist + int(cubies.corner_orient[new_slot])) % 3
                table[3 * old_slot + twist] = 3 * new_slot + new_twist
        corner_moves[move] = table
    return corner_moves


@cache
def _pair_distances(pair: int) -> dict[tuple[int, int], int]:
    """Fewest moves from every key to the solved pair."""
    solved = _pair_key(CubieCube.solved(), pair)
    distances = {solved: 0}
    queue = deque([solved])

    while queue:
        key = queue.popleft()
        for move, edge_table in _edge_moves().items():
            next_key = _move_pair_key(key, _corner_moves()[move], edge_table)
            if next_key not in distances:
                distances[next_key] = distances[key] + 1
                queue.append(next_key)

    return distances
