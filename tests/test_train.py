import pytest

torch = pytest.importorskip("torch")

from training.data import generate
from training.encoding import encode_stickers
from training.model import ValueNet
from training.train import (
    TrainConfig,
    load_checkpoint,
    save_checkpoint,
    train,
    train_epoch,
)

CPU = torch.device("cpu")
TINY = TrainConfig(epochs=2, cubes_per_epoch=64, batch_size=16, hidden_size=32)


def predict(model, seed=123):
    stickers, _ = generate(16, max_moves=10, seed=seed)
    with torch.no_grad():
        return model(encode_stickers(stickers))


def test_train_epoch_lowers_loss():
    config = TrainConfig(
        cubes_per_epoch=256, batch_size=32, max_moves=5, hidden_size=32
    )
    torch.manual_seed(0)
    model = ValueNet(hidden_size=config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = [train_epoch(model, optimizer, config, seed, CPU) for seed in range(10)]

    assert losses[-1] < losses[0] / 2


def test_train_uses_config(capsys):
    model = train(TINY, CPU)

    assert model.layers[0].out_features == 32
    assert "Epoch 2/2" in capsys.readouterr().out


def test_train_is_repeatable_with_a_seed():
    first = train(TINY, CPU)
    second = train(TINY, CPU)

    assert torch.equal(predict(first), predict(second))


def test_checkpoint_round_trip(tmp_path):
    model = train(TINY, CPU)
    path = tmp_path / "nested" / "value_net.pt"

    save_checkpoint(model, TINY, path)
    loaded = load_checkpoint(path, CPU)

    assert loaded.layers[0].out_features == 32
    assert not loaded.training
    assert torch.equal(predict(loaded), predict(model))
