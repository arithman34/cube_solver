import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cube.moves import MOVE_NAMES
from cube.state import Cube
from training.encoding import encode_stickers
from training.evaluate import (
    DepthResult,
    cubes_at_depth,
    evaluate,
    predict,
    print_results,
)
from training.model import ValueNet

CPU = torch.device("cpu")


class AlwaysThree(torch.nn.Module):
    def forward(self, x):
        return torch.full((len(x),), 3.0)


def small_model():
    torch.manual_seed(0)
    return ValueNet(hidden_size=32)


def test_cubes_at_depth_shape_and_dtype():
    stickers = cubes_at_depth(4, 10, np.random.default_rng(0))

    assert stickers.shape == (10, 54)
    assert stickers.dtype == np.uint8


def test_cubes_at_depth_one_gives_one_move_cubes():
    one_move_cubes = {
        Cube.solved().apply_move(move).stickers.tobytes() for move in MOVE_NAMES
    }

    stickers = cubes_at_depth(1, 50, np.random.default_rng(0))

    assert {row.tobytes() for row in stickers} <= one_move_cubes


def test_cubes_at_depth_two_is_never_solved_or_one_move():
    one_move_cubes = {
        Cube.solved().apply_move(move).stickers.tobytes() for move in MOVE_NAMES
    }

    stickers = cubes_at_depth(2, 200, np.random.default_rng(0))

    assert not {row.tobytes() for row in stickers} & one_move_cubes
    assert not (stickers == Cube.solved().stickers).all(axis=1).any()


def test_predict_matches_the_model():
    model = small_model()
    stickers = cubes_at_depth(5, 8, np.random.default_rng(0))

    predictions = predict(model, stickers, CPU)

    with torch.no_grad():
        expected = model(encode_stickers(stickers)).numpy()
    assert predictions.shape == (8,)
    assert predictions.dtype == np.float32
    assert np.array_equal(predictions, expected)


def test_predict_puts_model_in_eval_mode():
    model = small_model()
    model.train()

    predict(model, cubes_at_depth(3, 4, np.random.default_rng(0)), CPU)

    assert not model.training


def test_evaluate_with_a_constant_model():
    results = evaluate(AlwaysThree(), max_moves=5, n_per_depth=10, device=CPU)

    assert results == [
        DepthResult(depth=1, mean_prediction=3.0, mean_abs_error=2.0),
        DepthResult(depth=2, mean_prediction=3.0, mean_abs_error=1.0),
        DepthResult(depth=3, mean_prediction=3.0, mean_abs_error=0.0),
        DepthResult(depth=4, mean_prediction=3.0, mean_abs_error=1.0),
        DepthResult(depth=5, mean_prediction=3.0, mean_abs_error=2.0),
    ]


def test_evaluate_with_kociemba():
    results = evaluate(
        AlwaysThree(), max_moves=3, n_per_depth=10, device=CPU, kociemba_per_depth=2
    )

    assert results[0].kociemba_length == 1.0
    assert all(result.kociemba_length >= 1 for result in results)


def test_evaluate_is_repeatable_with_a_seed():
    model = small_model()

    first = evaluate(model, max_moves=4, n_per_depth=20, device=CPU, seed=7)
    second = evaluate(model, max_moves=4, n_per_depth=20, device=CPU, seed=7)
    other = evaluate(model, max_moves=4, n_per_depth=20, device=CPU, seed=8)

    assert first == second
    assert first != other


def test_print_results(capsys):
    print_results(
        [
            DepthResult(depth=1, mean_prediction=1.023, mean_abs_error=0.051),
            DepthResult(
                depth=20,
                mean_prediction=15.8,
                mean_abs_error=4.214,
                kociemba_length=23.1,
            ),
        ]
    )

    assert capsys.readouterr().out == (
        "depth  predicted  abs error  kociemba\n"
        "    1       1.02       0.05         -\n"
        "   20      15.80       4.21      23.1\n"
    )
