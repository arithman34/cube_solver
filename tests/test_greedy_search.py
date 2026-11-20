import pytest

torch = pytest.importorskip("torch")

from cube.moves import MOVE_NAMES
from cube.state import Cube
from training.search import SearchResult, greedy_search, neighbours

CPU = torch.device("cpu")


def decode(x):
    return x.view(len(x), 54, 6).argmax(dim=-1).to(torch.uint8).numpy()


class Oracle(torch.nn.Module):
    """Knows the distance of every cube along one scramble, and 99 for the rest."""

    def __init__(self, scramble):
        super().__init__()
        self.distances = {}
        cube = Cube.solved()
        for depth, move in enumerate(scramble.split(), start=1):
            cube = cube.apply_move(move)
            self.distances[cube.stickers.tobytes()] = depth

    def forward(self, x):
        values = [self.distances.get(row.tobytes(), 99) for row in decode(x)]
        return torch.tensor(values, dtype=torch.float32)


class Constant(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(len(x))


def test_neighbours_match_each_move():
    cube = Cube.solved().apply_moves("R U F'")

    next_cubes = neighbours(cube.stickers)

    assert next_cubes.shape == (18, 54)
    for move, stickers in zip(MOVE_NAMES, next_cubes, strict=True):
        assert (stickers == cube.apply_move(move).stickers).all()


def test_neighbours_of_solved_cube_are_never_solved():
    next_cubes = neighbours(Cube.solved().stickers)

    assert not any(Cube(row).is_solved() for row in next_cubes)


def test_greedy_search_on_solved_cube():
    result = greedy_search(Constant(), Cube.solved(), CPU)

    assert result == SearchResult(solved=True, moves=[], nodes=0)


def test_greedy_search_one_move_cube_needs_no_model():
    cube = Cube.solved().apply_move("R")

    result = greedy_search(Constant(), cube, CPU)

    assert result == SearchResult(solved=True, moves=["R'"], nodes=0)


def test_greedy_search_follows_the_model():
    scramble = "R U F' D2 L"
    cube = Cube.solved().apply_moves(scramble)

    result = greedy_search(Oracle(scramble), cube, CPU)

    assert result == SearchResult(
        solved=True, moves=["L'", "D2", "F", "U'", "R'"], nodes=18 * 4
    )
    assert cube.apply_moves(result.moves).is_solved()


def test_greedy_search_gives_up_after_max_steps():
    cube = Cube.scramble(20, seed=0)

    result = greedy_search(Constant(), cube, CPU, max_steps=5)

    assert not result.solved
    assert len(result.moves) == 5
    assert result.nodes == 18 * 5


def test_greedy_search_never_revisits_a_cube():
    cube = Cube.scramble(20, seed=1)

    result = greedy_search(Constant(), cube, CPU, max_steps=30)

    seen = {cube.stickers.tobytes()}
    for move in result.moves:
        cube = cube.apply_move(move)
        assert cube.stickers.tobytes() not in seen
        seen.add(cube.stickers.tobytes())
    assert len(seen) == 31
