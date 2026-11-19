import pytest

torch = pytest.importorskip("torch")

from cube.faces import Face
from cube.state import Cube
from training.encoding import encode, encode_batch

SOLVED = Cube.solved().to_string()


def test_encode_solved_cube():
    expected = torch.zeros(324)
    for i in range(54):
        expected[i * 6 + i // 9] = 1

    encoded = encode(SOLVED)

    assert encoded.dtype == torch.float32
    assert torch.equal(encoded, expected)


def test_encode_scrambled_cube_has_one_hot_per_sticker():
    cube = Cube.scramble(20, seed=7).to_string()

    stickers = encode(cube).view(54, 6)

    assert torch.equal(stickers.sum(dim=1), torch.ones(54))
    assert "".join(Face(i).name for i in stickers.argmax(dim=1).tolist()) == cube


def test_encode_rejects_wrong_length():
    with pytest.raises(ValueError):
        encode(SOLVED[:-1])


def test_encode_rejects_unknown_letter():
    with pytest.raises(ValueError, match="'X'"):
        encode("X" + SOLVED[1:])


def test_encode_batch_stacks_rows():
    cubes = [
        SOLVED,
        Cube.scramble(5, seed=1).to_string(),
        Cube.scramble(20, seed=2).to_string(),
    ]

    batch = encode_batch(cubes)

    assert batch.shape == (3, 324)
    for row, cube in zip(batch, cubes):
        assert torch.equal(row, encode(cube))
