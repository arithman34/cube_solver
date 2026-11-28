import pytest

from backend import solvers
from backend.solvers import (
    SolveFailed,
    SolverUnavailable,
    available_solvers,
    cfop_solver,
    kociemba_solver,
    neural_solver,
)
from cube.state import Cube


def test_kociemba_solver_solves_a_scramble():
    cube = Cube.scramble(20, seed=0)

    moves = kociemba_solver().solve(cube)

    assert cube.apply_moves(moves).is_solved()


def test_cfop_solver_solves_a_scramble():
    cube = Cube.scramble(20, seed=0)

    moves = cfop_solver().solve(cube, 1.0)

    assert cube.apply_moves(moves).is_solved()


def test_cfop_solver_ignores_the_time_limit():
    solver = cfop_solver()

    assert solver.name == "cfop"
    assert solver.label == "CFOP (human)"
    assert solver.uses_time_limit is False


def test_neural_solver_without_a_checkpoint(tmp_path):
    pytest.importorskip("torch")

    with pytest.raises(SolverUnavailable, match="No checkpoint"):
        neural_solver(tmp_path / "missing.pt")


def test_available_solvers_without_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(solvers, "MODEL_PATH", tmp_path / "missing.pt")

    found = available_solvers.__wrapped__()

    assert list(found) == ["kociemba", "cfop"]


@pytest.fixture
def small_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from training.model import build_model
    from training.train import save_checkpoint
    from training.value_iteration import ValueIterationConfig

    config = ValueIterationConfig(first_size=64, hidden_size=32, num_blocks=1)
    torch.manual_seed(0)
    path = tmp_path / "small.pt"
    save_checkpoint(build_model(vars(config)), config, path)
    return path


def test_neural_solver_loads_a_checkpoint(small_checkpoint):
    solver = neural_solver(small_checkpoint)

    assert solver.name == "neural"
    assert solver.label == "Neural (value iteration)"


def test_neural_solver_solves_a_one_move_cube(small_checkpoint):
    solver = neural_solver(small_checkpoint)

    assert solver.solve(Cube.solved().apply_move("R")) == ["R'"]


def test_neural_solver_with_an_untrained_model_gives_up(small_checkpoint):
    solver = neural_solver(small_checkpoint, beam_width=1)

    with pytest.raises(SolveFailed, match="Try Kociemba"):
        solver.solve(Cube.scramble(20, seed=0))


def test_available_solvers_with_a_checkpoint(small_checkpoint, monkeypatch):
    monkeypatch.setattr(solvers, "MODEL_PATH", small_checkpoint)

    found = available_solvers.__wrapped__()

    assert list(found) == ["kociemba", "cfop", "neural"]
