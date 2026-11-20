import pytest

torch = pytest.importorskip("torch")

from training.data import generate
from training.encoding import encode_stickers
from training.model import ResidualValueNet, ValueNet, build_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_predicts_one_distance_per_cube():
    stickers, _ = generate(8, max_moves=10, seed=0)

    predictions = ValueNet()(encode_stickers(stickers))

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
    stickers, depths = generate(32, max_moves=5, seed=1)
    x = encode_stickers(stickers)
    y = torch.from_numpy(depths).float()
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


def test_residual_net_predicts_one_float32_distance_per_cube():
    stickers, _ = generate(8, max_moves=10, seed=0)
    model = ResidualValueNet(first_size=64, hidden_size=32, num_blocks=2)

    predictions = model(encode_stickers(stickers))

    assert predictions.shape == (8,)
    assert predictions.dtype == torch.float32


def test_residual_net_output_stays_float32_under_autocast():
    stickers, _ = generate(8, max_moves=10, seed=0)
    model = ResidualValueNet(first_size=64, hidden_size=32, num_blocks=2).eval()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        predictions = model(encode_stickers(stickers))

    assert predictions.dtype == torch.float32


def test_residual_net_parameter_count():
    def linear(n_in, n_out):
        return n_in * n_out + n_out

    def batch_norm(n):
        return 2 * n

    block = 2 * (linear(1000, 1000) + batch_norm(1000))
    expected = (
        linear(324, 5000)
        + batch_norm(5000)
        + linear(5000, 1000)
        + batch_norm(1000)
        + 4 * block
        + linear(1000, 1)
    )

    assert count_parameters(ResidualValueNet()) == expected == 14_663_001


def test_residual_block_count():
    model = ResidualValueNet(first_size=64, hidden_size=32, num_blocks=3)

    assert len(model.blocks) == 3


def test_residual_net_reduces_loss_on_a_small_batch():
    torch.manual_seed(0)
    stickers, depths = generate(32, max_moves=5, seed=1)
    x = encode_stickers(stickers)
    y = torch.from_numpy(depths).float()
    model = ResidualValueNet(first_size=128, hidden_size=64, num_blocks=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] / 10


def test_build_model_defaults_to_the_mlp_for_supervised_configs():
    model = build_model({"hidden_size": 64})

    assert isinstance(model, ValueNet)
    assert model.layers[0].out_features == 64


def test_build_model_residual():
    model = build_model(
        {"model": "residual", "first_size": 64, "hidden_size": 32, "num_blocks": 2}
    )

    assert isinstance(model, ResidualValueNet)
    assert model.input[0].out_features == 64
    assert len(model.blocks) == 2
