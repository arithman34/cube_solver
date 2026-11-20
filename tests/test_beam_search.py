import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cube.state import Cube
from training.search import SearchResult, beam_search, expand, neighbours

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


def test_expand_matches_neighbours():
    beam = np.stack(
        [
            Cube.solved().stickers,
            Cube.solved().apply_moves("R U").stickers,
            Cube.scramble(10, seed=0).stickers,
        ]
    )

    children = expand(beam)

    assert children.shape == (54, 54)
    for i, child in enumerate(children):
        assert (child == neighbours(beam[i // 18])[i % 18]).all()


def test_beam_search_on_solved_cube():
    result = beam_search(Constant(), Cube.solved(), CPU)

    assert result == SearchResult(solved=True, moves=[], nodes=0)


def test_beam_search_one_move_cube_needs_no_model():
    cube = Cube.solved().apply_move("R")

    result = beam_search(Constant(), cube, CPU)

    assert result == SearchResult(solved=True, moves=["R'"], nodes=0)


def test_beam_search_follows_the_model():
    scramble = "R U F' D2 L"
    cube = Cube.solved().apply_moves(scramble)

    result = beam_search(Oracle(scramble), cube, CPU, beam_width=10)

    assert result.solved
    assert result.moves == ["L'", "D2", "F", "U'", "R'"]
    assert cube.apply_moves(result.moves).is_solved()


def test_beam_search_recovers_from_a_misleading_first_move():
    scramble = "R U F' D2 L"
    cube = Cube.solved().apply_moves(scramble)
    model = Oracle(scramble)
    model.distances[cube.apply_move("U").stickers.tobytes()] = 0.5

    narrow = beam_search(model, cube, CPU, beam_width=1, max_steps=10)
    wide = beam_search(model, cube, CPU, beam_width=2, max_steps=10)

    assert not narrow.solved
    assert wide.solved
    assert wide.moves == ["L'", "D2", "F", "U'", "R'"]


def test_beam_search_scores_each_cube_once():
    cube = Cube.scramble(20, seed=0)

    result = beam_search(Constant(), cube, CPU, beam_width=18, max_steps=2)

    assert result.nodes == 18 + 243


def test_beam_search_gives_up_after_max_steps():
    cube = Cube.scramble(20, seed=1)

    result = beam_search(Constant(), cube, CPU, beam_width=5, max_steps=3)

    assert not result.solved
    assert len(result.moves) == 3
    assert not cube.apply_moves(result.moves).is_solved()
