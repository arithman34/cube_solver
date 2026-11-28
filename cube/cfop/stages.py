from cube.cubies import CubieCube


def is_cross_solved(cubies: CubieCube) -> bool:
    """Whether the four D edges are home."""
    return all(_edge_solved(cubies, slot) for slot in range(4, 8))


def solved_pairs(cubies: CubieCube) -> list[int]:
    """The F2L pairs that are home. Pair i is corner 4 + i with edge 8 + i."""
    return [
        i
        for i in range(4)
        if _corner_solved(cubies, 4 + i) and _edge_solved(cubies, 8 + i)
    ]


def is_f2l_solved(cubies: CubieCube) -> bool:
    """Whether the cross and all four pairs are home."""
    return is_cross_solved(cubies) and len(solved_pairs(cubies)) == 4


def is_oll_solved(cubies: CubieCube) -> bool:
    """Whether F2L is solved and the U face is all one color."""
    return (
        is_f2l_solved(cubies)
        and not any(cubies.corner_orient[:4])
        and not any(cubies.edge_orient[:4])
    )


def is_pll_solved(cubies: CubieCube) -> bool:
    """Whether the whole cube is solved, including the last turn of U."""
    return cubies == CubieCube.solved()


def _corner_solved(cubies: CubieCube, slot: int) -> bool:
    """Whether the corner in slot is home and not twisted."""
    return bool(cubies.corner_perm[slot] == slot and cubies.corner_orient[slot] == 0)


def _edge_solved(cubies: CubieCube, slot: int) -> bool:
    """Whether the edge in slot is home and not flipped."""
    return bool(cubies.edge_perm[slot] == slot and cubies.edge_orient[slot] == 0)
