from collections import deque
from functools import cache

from cube.cubies import MOVE_CUBIES, NUM_EDGES, CubieCube


def _cross_key(cubies: CubieCube) -> tuple[int, ...]:
    """Where the four D edges are, as 2 * slot + flip for each."""
    key = []
    for edge in range(4, 8):
        slot = list(cubies.edge_perm).index(edge)
        flip = int(cubies.edge_orient[slot])
        key.append(2 * slot + flip)

    return tuple(key)


@cache
def _edge_moves() -> dict[str, list[int]]:
    """For each move, table[position] = where an edge at that position ends up."""
    edge_moves = {}
    for move, cubies in MOVE_CUBIES.items():
        table = [0] * (2 * NUM_EDGES)
        for new_slot in range(NUM_EDGES):
            old_slot = int(cubies.edge_perm[new_slot])
            for flip in range(2):
                new_flip = (flip + int(cubies.edge_orient[new_slot])) % 2
                table[2 * old_slot + flip] = 2 * new_slot + new_flip
        edge_moves[move] = table
    return edge_moves


def _move_key(key: tuple[int, ...], table: list[int]) -> tuple[int, ...]:
    """The key after the move whose table is given."""
    return tuple(table[position] for position in key)


@cache
def _distances() -> dict[tuple[int, ...], int]:
    """Fewest moves from every key to the solved cross, by breadth-first search."""
    solved = _cross_key(CubieCube.solved())
    distances = {solved: 0}
    queue = deque([solved])

    while queue:
        key = queue.popleft()
        for table in _edge_moves().values():
            next_key = _move_key(key, table)
            if next_key not in distances:
                distances[next_key] = distances[key] + 1
                queue.append(next_key)

    return distances


def solve_cross(cubies: CubieCube) -> list[str]:
    """Shortest moves that solve the cross on D."""
    edge_moves = _edge_moves()
    distances = _distances()

    key = _cross_key(cubies)
    moves = []
    while distances[key] > 0:
        for move, table in edge_moves.items():
            next_key = _move_key(key, table)
            if distances[next_key] < distances[key]:
                moves.append(move)
                key = next_key
                break

    return moves
