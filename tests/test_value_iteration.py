import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cube.moves import MOVE_NAMES
from cube.state import Cube
from training.data import generate
from training.evaluate import predict
from training.model import ResidualValueNet, ValueNet
from training.train import load_checkpoint, save_checkpoint
from training.value_iteration import (
    ValueIterationConfig,
    bellman_targets,
    decay_rate,
    should_update_target,
    value_iteration,
    value_iteration_step,
)

CPU = torch.device("cpu")
SMALL = {"first_size": 64, "hidden_size": 32, "num_blocks": 1}
TINY = ValueIterationConfig(
    iterations=2, cubes_per_iteration=64, batch_size=16, max_moves=5, **SMALL
)


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
        return torch.full((len(x),), 5.0)


def stickers_of(*scrambles):
    return np.stack([Cube.solved().apply_moves(s).stickers for s in scrambles])


def test_bellman_targets_with_a_constant_model():
    stickers = stickers_of("R", "U2", "R U F", "L D")

    targets = bellman_targets(Constant(), stickers, CPU)

    assert targets.tolist() == [1.0, 1.0, 6.0, 6.0]
    assert targets.dtype == np.float32


def test_bellman_targets_take_the_closest_neighbour():
    stickers = stickers_of("R", "R U", "R U F'", "R U F' D2")

    targets = bellman_targets(Oracle("R U F' D2"), stickers, CPU)

    assert targets.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_bellman_targets_solved_cube_is_zero():
    stickers = stickers_of("", "R")

    targets = bellman_targets(Constant(), stickers, CPU)

    assert targets.tolist() == [0.0, 1.0]


def test_bellman_targets_same_in_chunks():
    torch.manual_seed(0)
    model = ValueNet(hidden_size=32)
    stickers, _ = generate(25, max_moves=10, seed=0)

    whole = bellman_targets(model, stickers, CPU, chunk_size=1_000)
    chunked = bellman_targets(model, stickers, CPU, chunk_size=7)

    assert chunked.shape == (25,)
    assert np.allclose(chunked, whole)


def test_value_iteration_step_fits_fixed_targets():
    config = ValueIterationConfig(
        cubes_per_iteration=256, batch_size=32, max_moves=5, hidden_size=32
    )
    torch.manual_seed(0)
    model = ValueNet(hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    steps = [
        value_iteration_step(model, Constant(), optimizer, config, seed, CPU)
        for seed in range(10)
    ]
    losses = [loss for loss, _ in steps]

    assert losses[-1] < losses[0] / 10


def test_value_iteration_step_reports_the_mean_target():
    config = ValueIterationConfig(
        cubes_per_iteration=100, batch_size=50, max_moves=5, hidden_size=32
    )
    model = ValueNet(hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters())
    stickers, _ = generate(100, max_moves=5, seed=0)
    one_move = (bellman_targets(Constant(), stickers, CPU) == 1).sum()

    _, mean_target = value_iteration_step(model, Constant(), optimizer, config, 0, CPU)

    assert mean_target == pytest.approx((one_move * 1 + (100 - one_move) * 6) / 100)


@pytest.mark.parametrize(
    "loss, since_update, expected",
    [
        (0.01, 1, True),
        (0.2, 1, False),
        (0.2, 9, False),
        (0.2, 10, True),
        (0.05, 3, False),
    ],
)
def test_should_update_target(loss, since_update, expected):
    config = ValueIterationConfig(update_threshold=0.05, update_every=10)

    assert should_update_target(loss, since_update, config) is expected


def test_value_iteration_step_trains_a_residual_net():
    config = ValueIterationConfig(
        cubes_per_iteration=256, batch_size=32, max_moves=5, **SMALL
    )
    torch.manual_seed(0)
    model = ResidualValueNet(**SMALL)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    steps = [
        value_iteration_step(model, Constant(), optimizer, config, seed, CPU)
        for seed in range(10)
    ]
    losses = [loss for loss, _ in steps]

    assert losses[-1] < losses[0] / 10


def test_value_iteration_uses_config(capsys):
    model = value_iteration(TINY, CPU)

    assert isinstance(model, ResidualValueNet)
    assert model.input[0].out_features == 64
    assert len(model.blocks) == 1
    assert model.output.in_features == 32
    assert "Iteration 2/2" in capsys.readouterr().out


def test_value_iteration_can_train_the_mlp():
    config = ValueIterationConfig(
        iterations=1, cubes_per_iteration=64, batch_size=16, model="mlp", hidden_size=32
    )

    model = value_iteration(config, CPU)

    assert isinstance(model, ValueNet)
    assert model.layers[0].out_features == 32


def test_value_iteration_reports_target_updates(capsys):
    config = ValueIterationConfig(
        iterations=3,
        cubes_per_iteration=64,
        batch_size=16,
        update_threshold=0,
        update_every=2,
        **SMALL,
    )

    value_iteration(config, CPU)

    lines = capsys.readouterr().out.splitlines()
    assert "target update" not in lines[0]
    assert lines[1].endswith("target update 1")
    assert "target update" not in lines[2]


def test_value_iteration_is_repeatable_with_a_seed():
    first = value_iteration(TINY, CPU)
    second = value_iteration(TINY, CPU)
    stickers, _ = generate(16, max_moves=5, seed=123)

    assert np.array_equal(predict(first, stickers, CPU), predict(second, stickers, CPU))


def test_value_iteration_learns_short_distances_from_nothing():
    config = ValueIterationConfig(
        iterations=40,
        cubes_per_iteration=500,
        batch_size=50,
        max_moves=2,
        model="mlp",
        hidden_size=64,
        learning_rate=3e-3,
        update_threshold=0,
        update_every=4,
    )
    one_move = stickers_of(*MOVE_NAMES)
    two_moves = stickers_of("R U", "F' L2", "D B", "U2 R'", "L F")

    model = value_iteration(config, CPU)

    assert np.abs(predict(model, one_move, CPU) - 1).max() < 0.1
    assert np.abs(predict(model, two_moves, CPU) - 2).max() < 0.1


def test_residual_value_iteration_learns_short_distances_from_nothing():
    config = ValueIterationConfig(
        iterations=40,
        cubes_per_iteration=1_000,
        batch_size=100,
        max_moves=2,
        first_size=128,
        hidden_size=64,
        num_blocks=1,
        update_every=4,
    )
    one_move = stickers_of(*MOVE_NAMES)
    two_moves = stickers_of("R U", "F' L2", "D B", "U2 R'", "L F")

    model = value_iteration(config, CPU)

    assert np.abs(predict(model, one_move, CPU) - 1).max() < 0.3
    assert np.abs(predict(model, two_moves, CPU) - 2).max() < 0.3


def test_checkpoint_round_trip(tmp_path):
    model = value_iteration(TINY, CPU)
    path = tmp_path / "value_iteration.pt"
    stickers, _ = generate(16, max_moves=5, seed=123)

    save_checkpoint(model, TINY, path)
    loaded = load_checkpoint(path, CPU)

    assert isinstance(loaded, ResidualValueNet)
    assert not loaded.training
    assert np.array_equal(predict(loaded, stickers, CPU), predict(model, stickers, CPU))


def test_value_iteration_saves_every_few_updates(tmp_path, capsys):
    config = ValueIterationConfig(
        iterations=3,
        cubes_per_iteration=64,
        batch_size=16,
        update_threshold=1e9,
        save_every=2,
        **SMALL,
    )
    path = tmp_path / "value_iteration.pt"

    model = value_iteration(config, CPU, path)

    lines = capsys.readouterr().out.splitlines()
    assert [line.endswith("saved") for line in lines] == [False, True, False]
    assert path.exists()
    stickers, _ = generate(16, max_moves=5, seed=123)
    assert not np.array_equal(
        predict(load_checkpoint(path, CPU), stickers, CPU),
        predict(model, stickers, CPU),
    )


def test_value_iteration_saves_nothing_without_a_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = ValueIterationConfig(
        iterations=2,
        cubes_per_iteration=64,
        batch_size=16,
        update_threshold=1e9,
        save_every=1,
        **SMALL,
    )

    value_iteration(config, CPU)

    assert list(tmp_path.iterdir()) == []


def test_decay_rate_reaches_the_final_learning_rate_on_the_last_iteration():
    config = ValueIterationConfig(
        iterations=400, learning_rate=1e-3, final_learning_rate=1e-4
    )

    rate = decay_rate(config)

    assert 1e-3 * rate**399 == pytest.approx(1e-4)


def test_decay_rate_for_a_single_iteration():
    assert decay_rate(ValueIterationConfig(iterations=1)) == 1.0


def test_value_iteration_decays_the_learning_rate(capsys):
    config = ValueIterationConfig(
        iterations=3,
        cubes_per_iteration=64,
        batch_size=16,
        learning_rate=1e-2,
        final_learning_rate=1e-4,
        **SMALL,
    )

    value_iteration(config, CPU)

    lines = capsys.readouterr().out.splitlines()
    assert "lr 1.0e-02" in lines[0]
    assert "lr 1.0e-03" in lines[1]
    assert "lr 1.0e-04" in lines[2]
