import pytest
from fastapi.testclient import TestClient

from backend.main import app
from cube.cubies import CORNER_FACELETS, EDGE_FACELETS
from cube.state import Cube
from cube.validation import CubeError

SOLVED = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def _swap(cube: str, dst: list[int], src: list[int]) -> str:
    letters = list(cube)
    for d, s in zip(dst, src, strict=True):
        letters[d] = cube[s]
    return "".join(letters)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"health": "ok"}


def test_solve_solved_cube(client):
    response = client.post("/solve", json={"cube": SOLVED})
    assert response.status_code == 200
    assert response.json() == {"moves": [], "length": 0}


def test_solve_one_move(client):
    cube = Cube.solved().apply_move("R").to_string()
    response = client.post("/solve", json={"cube": cube})
    assert response.json() == {"moves": ["R'"], "length": 1}


@pytest.mark.parametrize("seed", range(3))
def test_solve_scrambled_cube(client, seed):
    cube = Cube.scramble(30, seed=seed)
    response = client.post("/solve", json={"cube": cube.to_string()})
    body = response.json()
    assert response.status_code == 200
    assert cube.apply_moves(body["moves"]).is_solved()
    assert body["length"] == len(body["moves"])


@pytest.mark.parametrize(
    "cube",
    ["", "X", SOLVED[:-1], SOLVED + "U", SOLVED.lower(), "X" + SOLVED[1:]],
)
def test_solve_rejects_badly_formatted_cube(client, cube):
    response = client.post("/solve", json={"cube": cube})
    assert response.status_code == 422


def test_solve_rejects_missing_cube(client):
    assert client.post("/solve", json={}).status_code == 422


def test_solve_reports_flipped_edge(client):
    edge = EDGE_FACELETS[0].tolist()
    cube = _swap(SOLVED, edge, edge[::-1])
    response = client.post("/solve", json={"cube": cube})
    assert response.status_code == 422
    assert response.json() == {"detail": [CubeError.FLIPPED_EDGE.value]}


def test_solve_reports_every_error(client):
    corner = CORNER_FACELETS[0].tolist()
    edge = EDGE_FACELETS[0].tolist()
    cube = _swap(SOLVED, corner, corner[1:] + corner[:1])
    cube = _swap(cube, edge, edge[::-1])
    response = client.post("/solve", json={"cube": cube})
    assert response.json() == {
        "detail": [CubeError.TWISTED_CORNER.value, CubeError.FLIPPED_EDGE.value]
    }


def test_solve_reports_wrong_sticker_counts(client):
    response = client.post("/solve", json={"cube": "D" + SOLVED[1:]})
    assert response.json() == {"detail": [CubeError.BAD_COUNTS.value]}


def test_solve_is_post_only(client):
    assert client.get("/solve").status_code == 405


def test_scramble_default_is_25_moves(client):
    response = client.get("/scramble")
    assert response.status_code == 200
    assert len(response.json()["moves"]) == 25


def test_scramble_moves_make_the_cube(client):
    body = client.get("/scramble", params={"num_moves": 20, "seed": 1}).json()
    assert Cube.solved().apply_moves(body["moves"]).to_string() == body["cube"]
    assert body["cube"] == Cube.scramble(20, seed=1).to_string()


def test_scramble_same_seed_same_cube(client):
    first = client.get("/scramble", params={"seed": 7}).json()
    second = client.get("/scramble", params={"seed": 7}).json()
    assert first == second


def test_scramble_zero_moves_is_solved(client):
    body = client.get("/scramble", params={"num_moves": 0}).json()
    assert body == {"cube": SOLVED, "moves": []}


@pytest.mark.parametrize("num_moves, status", [(-1, 422), (100, 200), (101, 422)])
def test_scramble_move_limits(client, num_moves, status):
    response = client.get("/scramble", params={"num_moves": num_moves})
    assert response.status_code == status


def test_scramble_then_solve(client):
    cube = client.get("/scramble", params={"num_moves": 30, "seed": 3}).json()["cube"]
    moves = client.post("/solve", json={"cube": cube}).json()["moves"]
    assert Cube.from_string(cube).apply_moves(moves).is_solved()
