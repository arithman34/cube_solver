import numpy as np
import pytest

from cube.cubies import CubieCube
from cube.kociemba import tables
from cube.kociemba.coords import (
    get_corners,
    get_flip,
    get_slice,
    get_slice_perm,
    get_twist,
    get_ud_edges,
)
from cube.kociemba.tables import PHASE_1_MOVES, PHASE_2_MOVES, get_tables

PHASE_1 = [
    ("twist_move", get_twist),
    ("flip_move", get_flip),
    ("slice_move", get_slice),
]
PHASE_2 = [
    ("corners_move", get_corners),
    ("ud_edges_move", get_ud_edges),
    ("slice_perm_move", get_slice_perm),
]
PRUNING = [
    ("twist_slice_prune", "twist_move", "slice_move"),
    ("flip_slice_prune", "flip_move", "slice_move"),
    ("corners_slice_perm_prune", "corners_move", "slice_perm_move"),
    ("ud_edges_slice_perm_prune", "ud_edges_move", "slice_perm_move"),
]


def _move(move: str) -> int:
    return PHASE_1_MOVES.index(move)


def test_phase_2_moves_are_all_real_moves():
    assert set(PHASE_2_MOVES) <= set(PHASE_1_MOVES)
    assert len(PHASE_2_MOVES) == 10


def test_move_table_shapes():
    t = get_tables()
    assert t.twist_move.shape == (2187, 18)
    assert t.flip_move.shape == (2048, 18)
    assert t.slice_move.shape == (495, 18)
    assert t.corners_move.shape == (40320, 10)
    assert t.ud_edges_move.shape == (40320, 10)
    assert t.slice_perm_move.shape == (24, 10)


def test_one_move_from_solved():
    t = get_tables()
    assert (t.twist_move[0, _move("R")], t.slice_move[0, _move("R")]) == (110, 367)
    assert (t.twist_move[0, _move("F")], t.flip_move[0, _move("F")]) == (412, 802)
    assert t.slice_move[0, _move("F")] == 230
    assert t.corners_move[0, PHASE_2_MOVES.index("U")] == 28560


@pytest.mark.parametrize("name, _", PHASE_1 + PHASE_2)
def test_every_move_column_is_a_permutation(name, _):
    table = getattr(get_tables(), name)
    size = table.shape[0]
    for m in range(table.shape[1]):
        assert sorted(table[:, m].tolist()) == list(range(size))


@pytest.mark.parametrize("name, _", PHASE_1)
def test_move_then_its_inverse_returns(name, _):
    table = getattr(get_tables(), name)
    values = np.arange(table.shape[0])
    for face in "URFDLB":
        there = table[values, _move(face)]
        assert np.array_equal(table[there, _move(face + "'")], values)


@pytest.mark.parametrize("seed", range(3))
def test_phase_1_tables_follow_a_scramble(seed):
    t = get_tables()
    rng = np.random.default_rng(seed)
    cubies = CubieCube.solved()
    coords = {name: 0 for name, _ in PHASE_1}
    for m in rng.integers(len(PHASE_1_MOVES), size=40):
        cubies = cubies.apply_move(PHASE_1_MOVES[m])
        coords = {name: getattr(t, name)[value, m] for name, value in coords.items()}
    assert coords == {name: get(cubies) for name, get in PHASE_1}


@pytest.mark.parametrize("seed", range(3))
def test_phase_2_tables_follow_a_scramble(seed):
    t = get_tables()
    rng = np.random.default_rng(seed)
    cubies = CubieCube.solved()
    coords = {name: 0 for name, _ in PHASE_2}
    for m in rng.integers(len(PHASE_2_MOVES), size=40):
        cubies = cubies.apply_move(PHASE_2_MOVES[m])
        coords = {name: getattr(t, name)[value, m] for name, value in coords.items()}
    assert coords == {name: get(cubies) for name, get in PHASE_2}


@pytest.mark.parametrize(
    "name, depths",
    [
        (
            "twist_slice_prune",
            [1, 4, 50, 586, 6184, 54066, 296322, 582492, 142520, 340],
        ),
        ("flip_slice_prune", [1, 4, 50, 532, 4804, 41832, 238263, 555636, 172314, 324]),
        (
            "ud_edges_slice_perm_prune",
            [
                1,
                10,
                67,
                456,
                3063,
                18202,
                86691,
                290812,
                434814,
                120488,
                11818,
                1114,
                144,
            ],
        ),
    ],
)
def test_pruning_depth_counts(name, depths):
    prune = getattr(get_tables(), name)
    assert np.bincount(prune.ravel()).tolist() == depths


@pytest.mark.parametrize("prune_name, a_name, b_name", PRUNING)
def test_pruning_table_is_complete(prune_name, a_name, b_name):
    t = get_tables()
    prune = getattr(t, prune_name)
    assert prune.dtype == np.int8
    assert prune.shape == (getattr(t, a_name).shape[0], getattr(t, b_name).shape[0])
    assert prune[0, 0] == 0
    assert (prune == -1).sum() == 0


@pytest.mark.parametrize("prune_name, a_name, b_name", PRUNING)
def test_one_move_changes_pruning_depth_by_at_most_one(prune_name, a_name, b_name):
    t = get_tables()
    prune = getattr(t, prune_name).astype(int)
    move_a, move_b = getattr(t, a_name), getattr(t, b_name)
    for m in range(move_a.shape[1]):
        after = prune[np.ix_(move_a[:, m], move_b[:, m])]
        assert np.abs(after - prune).max() <= 1


def test_pruning_after_one_move():
    t = get_tables()
    assert t.twist_slice_prune[110, 367] == 1
    assert t.flip_slice_prune[802, 230] == 1
    assert t.corners_slice_perm_prune[28560, 0] == 1


@pytest.mark.parametrize("seed", range(5))
def test_pruning_never_overestimates_a_scramble(seed):
    t = get_tables()
    cubies = CubieCube.solved()
    rng = np.random.default_rng(seed)
    for move in rng.choice(PHASE_1_MOVES, size=6):
        cubies = cubies.apply_move(str(move))
    assert t.twist_slice_prune[get_twist(cubies), get_slice(cubies)] <= 6
    assert t.flip_slice_prune[get_flip(cubies), get_slice(cubies)] <= 6


def test_load_or_build_saves_then_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "DATA_DIR", tmp_path / "data")
    calls = []

    def build():
        calls.append(1)
        return np.arange(5, dtype=np.int32)

    first = tables._load_or_build("example", build)
    second = tables._load_or_build("example", build)
    assert (tmp_path / "data" / "example.npy").exists()
    assert first.tolist() == second.tolist() == [0, 1, 2, 3, 4]
    assert second.dtype == np.int32
    assert len(calls) == 1


def test_get_tables_is_cached():
    assert get_tables() is get_tables()
