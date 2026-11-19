import torch
import torch.nn.functional as F
from torch import Tensor

from cube.faces import NUM_FACES, TOTAL_STICKERS, Face

FACES = {face.name: face.value for face in Face}


def encode(cube: str) -> Tensor:
    """One-hot encode a 54-letter cube into a float tensor of shape (324,).

    Sticker i with face f sets index i * 6 + f.
    """
    if len(cube) != TOTAL_STICKERS:
        raise ValueError(
            f"Cube must be {TOTAL_STICKERS} characters long. Got {len(cube)}."
        )

    try:
        indices = torch.tensor([FACES[letter] for letter in cube])
    except KeyError as error:
        raise ValueError(
            f"Unknown face letter {error.args[0]!r}. Expected one of {', '.join(FACES)}."
        ) from None

    return F.one_hot(indices, num_classes=NUM_FACES).flatten().float()


def encode_batch(cubes: list[str]) -> Tensor:
    """Encode several cubes into a tensor of shape (n, 324)."""
    return torch.stack([encode(cube) for cube in cubes])
