import pytest

from cube.cubies import CubieCube
from cube.kociemba.coords import (
    N_CORNERS,
    N_FLIP,
    N_SLICE,
    N_SLICE_PERM,
    N_TWIST,
    N_UD_EDGES,
    _rank_permutation,
    _unrank_permutation,
    get_corners,
    get_flip,
    get_slice,
    get_slice_perm,
    get_twist,
    get_ud_edges,
    set_corners,
    set_flip,
    set_slice,
    set_slice_perm,
    set_twist,
    set_ud_edges,
)

COORDINATES = [
    (get_twist, set_twist, N_TWIST),
    (get_flip, set_flip, N_FLIP),
    (get_slice, set_slice, N_SLICE),
    (get_corners, set_corners, N_CORNERS),
    (get_ud_edges, set_ud_edges, N_UD_EDGES),
    (get_slice_perm, set_slice_perm, N_SLICE_PERM),
]


def _cubies(moves: str) -> CubieCube:
    cubies = CubieCube.solved()
    for move in moves.split():
        cubies = cubies.apply_move(move)
    return cubies


def test_sizes():
    assert (N_TWIST, N_FLIP, N_SLICE) == (2187, 2048, 495)
    assert (N_CORNERS, N_UD_EDGES, N_SLICE_PERM) == (40320, 40320, 24)


@pytest.mark.parametrize(
    "perm, rank",
    [([0, 1, 2], 0), ([2, 0, 1], 4), ([2, 1, 0], 5), ([3, 2, 1, 0], 23)],
)
def test_rank_permutation(perm, rank):
    assert _rank_permutation(perm) == rank
    assert _unrank_permutation(rank, len(perm)).tolist() == perm


@pytest.mark.parametrize("get, set_, size", COORDINATES)
def test_solved_is_zero(get, set_, size):
    assert get(CubieCube.solved()) == 0
    assert set_(0) == CubieCube.solved()


@pytest.mark.parametrize("get, set_, size", COORDINATES)
def test_round_trip_every_value(get, set_, size):
    assert [get(set_(value)) for value in range(size)] == list(range(size))


@pytest.mark.parametrize("get, set_, size", COORDINATES)
def test_set_builds_a_real_cube(get, set_, size):
    for value in range(0, size, max(1, size // 100)):
        cubies = set_(value)
        assert sorted(cubies.corner_perm.tolist()) == list(range(8))
        assert sorted(cubies.edge_perm.tolist()) == list(range(12))
        assert cubies.corner_orient.sum() % 3 == 0
        assert cubies.edge_orient.sum() % 2 == 0


@pytest.mark.parametrize(
    "moves, twist, flip, slice_",
    [("R", 110, 0, 367), ("F", 412, 802, 230), ("U", 0, 0, 0)],
)
def test_phase_1_coordinates_after_one_move(moves, twist, flip, slice_):
    cubies = _cubies(moves)
    assert (get_twist(cubies), get_flip(cubies), get_slice(cubies)) == (
        twist,
        flip,
        slice_,
    )


def test_phase_2_moves_keep_phase_1_coordinates_at_zero():
    cubies = _cubies("R2 U F2 D' L2 B2 U2 D R2 U'")
    assert (get_twist(cubies), get_flip(cubies), get_slice(cubies)) == (0, 0, 0)
    assert get_corners(cubies) != 0


def test_slice_ignores_order_of_slice_edges():
    cubies = _cubies("R2 L2")
    assert get_slice(cubies) == 0
    assert get_slice_perm(cubies) != 0


def test_u_move_changes_only_permutation_coordinates():
    cubies = _cubies("U")
    assert get_corners(cubies) == 28560
    assert get_ud_edges(cubies) == 28560
    assert get_slice_perm(cubies) == 0
