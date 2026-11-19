import pytest

torch = pytest.importorskip("torch")

from training.data import generate
from training.encoding import encode_batch
from training.model import ValueNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_predicts_one_distance_per_cube():
    cubes, _ = generate(8, max_moves=10, seed=0)

    predictions = ValueNet()(encode_batch(cubes))

    assert predictions.shape == (8,)
    assert predictions.dtype == torch.float32


def test_single_cube_batch_keeps_batch_dimension():
    predictions = ValueNet()(torch.zeros(1, 324))

    assert predictions.shape == (1,)


def test_parameter_count():
    assert count_parameters(ValueNet()) == 429_569
    assert (
        count_parameters(ValueNet(hidden_size=64))
        == 324 * 64 + 64 + 64 * 64 + 64 + 64 + 1
    )


def test_training_reduces_loss_on_a_small_batch():
    torch.manual_seed(0)
    cubes, depths = generate(32, max_moves=5, seed=1)
    x = encode_batch(cubes)
    y = torch.tensor(depths, dtype=torch.float32)
    model = ValueNet(hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] / 10
