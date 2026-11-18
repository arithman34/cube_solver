import pytest

from cube.cubies import CORNER_FACELETS, EDGE_FACELETS
from cube.faces import Face, center_index
from cube.state import Cube
from cube.validation import CubeError, find_errors, is_solvable

URF, UFL = CORNER_FACELETS[0].tolist(), CORNER_FACELETS[1].tolist()
UR, UF = EDGE_FACELETS[0].tolist(), EDGE_FACELETS[1].tolist()


def _move_stickers(cube: Cube, dst: list[int], src: list[int]) -> Cube:
    stickers = cube.to_array()
    stickers[dst] = cube.stickers[src]
    return Cube(stickers)


def _twist_corner(cube: Cube) -> Cube:
    return _move_stickers(cube, URF, URF[1:] + URF[:1])


def _flip_edge(cube: Cube) -> Cube:
    return _move_stickers(cube, UR, UR[::-1])


def _swap_edges(cube: Cube) -> Cube:
    return _move_stickers(cube, UR + UF, UF + UR)


def _swap_corners(cube: Cube) -> Cube:
    return _move_stickers(cube, URF + UFL, UFL + URF)


def test_solved_cube_has_no_errors():
    assert find_errors(Cube.solved()) == []
    assert is_solvable(Cube.solved())


@pytest.mark.parametrize("seed", range(5))
def test_scrambled_cube_is_solvable(seed):
    assert find_errors(Cube.scramble(30, seed=seed)) == []


def test_bad_centers():
    cube = _move_stickers(
        Cube.solved(),
        [center_index(Face.U), center_index(Face.D)],
        [center_index(Face.D), center_index(Face.U)],
    )
    assert find_errors(cube) == [CubeError.BAD_CENTERS]


def test_bad_counts():
    stickers = Cube.solved().to_array()
    stickers[0] = Face.D
    assert find_errors(Cube(stickers)) == [CubeError.BAD_COUNTS]


def test_piece_that_does_not_exist():
    cube = _move_stickers(Cube.solved(), [8, 10], [10, 8])
    assert find_errors(cube) == [CubeError.INVALID_PIECE]


@pytest.mark.parametrize(
    "break_cube, expected",
    [
        (_twist_corner, CubeError.TWISTED_CORNER),
        (_flip_edge, CubeError.FLIPPED_EDGE),
        (_swap_edges, CubeError.SWAPPED_PIECES),
        (_swap_corners, CubeError.SWAPPED_PIECES),
    ],
)
def test_single_error_on_scrambled_cube(break_cube, expected):
    cube = break_cube(Cube.scramble(30, seed=3))
    assert find_errors(cube) == [expected]
    assert not is_solvable(cube)


def test_twisted_corner_and_flipped_edge_are_both_reported():
    cube = _flip_edge(_twist_corner(Cube.solved()))
    assert find_errors(cube) == [CubeError.TWISTED_CORNER, CubeError.FLIPPED_EDGE]
