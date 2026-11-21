import pytest
from fastapi.testclient import TestClient

from backend import routes
from backend.main import app
from backend.solvers import SolveFailed, Solver
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


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="cube-net"' in response.text


def test_home_page_links_static_files(client):
    page = client.get("/").text
    assert 'href="/static/style.css"' in page
    assert 'src="/static/app.js"' in page


@pytest.mark.parametrize(
    "path, content_type",
    [("/static/app.js", "text/javascript"), ("/static/style.css", "text/css")],
)
def test_static_files_are_served(client, path, content_type):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)


@pytest.mark.parametrize("path", ["/does-not-exist", "/static/nope.js", "/solve"])
def test_unknown_path_is_not_found(client, path):
    assert client.get(path).status_code == 404


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"health": "ok"}


def test_solve_solved_cube(client):
    response = client.post("/api/solve", json={"cube": SOLVED})
    body = response.json()
    assert response.status_code == 200
    assert body["moves"] == []
    assert body["length"] == 0
    assert body["method"] == "kociemba"


def test_solve_one_move(client):
    cube = Cube.solved().apply_move("R").to_string()
    response = client.post("/api/solve", json={"cube": cube})
    body = response.json()
    assert body["moves"] == ["R'"]
    assert body["length"] == 1
    assert body["time_ms"] >= 0


def test_solve_with_kociemba_by_name(client):
    cube = Cube.solved().apply_moves(["R", "U"]).to_string()
    response = client.post("/api/solve", json={"cube": cube, "method": "kociemba"})
    body = response.json()
    assert response.status_code == 200
    assert body["moves"] == ["U'", "R'"]
    assert body["method"] == "kociemba"


def test_solve_with_unknown_method(client):
    response = client.post("/api/solve", json={"cube": SOLVED, "method": "magic"})
    assert response.status_code == 422
    [message] = response.json()["detail"]
    assert message.startswith("Unknown method 'magic'. Choose from: kociemba")


def test_methods_lists_kociemba(client):
    response = client.get("/api/methods")
    assert response.status_code == 200
    assert {"name": "kociemba", "label": "Kociemba (two-phase)"} in response.json()


def test_solve_reports_a_failed_search(client, monkeypatch):
    def give_up(cube):
        raise SolveFailed("No solution found.")

    stuck = Solver(name="stuck", label="Always gives up", solve=give_up)
    monkeypatch.setattr(routes, "available_solvers", lambda: {"stuck": stuck})

    cube = Cube.solved().apply_move("R").to_string()
    response = client.post("/api/solve", json={"cube": cube, "method": "stuck"})

    assert response.status_code == 422
    assert response.json() == {"detail": ["No solution found."]}


@pytest.mark.parametrize("seed", range(3))
def test_solve_scrambled_cube(client, seed):
    cube = Cube.scramble(30, seed=seed)
    response = client.post("/api/solve", json={"cube": cube.to_string()})
    body = response.json()
    assert response.status_code == 200
    assert cube.apply_moves(body["moves"]).is_solved()
    assert body["length"] == len(body["moves"])


@pytest.mark.parametrize(
    "cube",
    ["", "X", SOLVED[:-1], SOLVED + "U", SOLVED.lower(), "X" + SOLVED[1:]],
)
def test_solve_rejects_badly_formatted_cube(client, cube):
    response = client.post("/api/solve", json={"cube": cube})
    assert response.status_code == 422


def test_solve_rejects_missing_cube(client):
    assert client.post("/api/solve", json={}).status_code == 422


def test_solve_reports_flipped_edge(client):
    edge = EDGE_FACELETS[0].tolist()
    cube = _swap(SOLVED, edge, edge[::-1])
    response = client.post("/api/solve", json={"cube": cube})
    assert response.status_code == 422
    assert response.json() == {"detail": [CubeError.FLIPPED_EDGE.value]}


def test_solve_reports_every_error(client):
    corner = CORNER_FACELETS[0].tolist()
    edge = EDGE_FACELETS[0].tolist()
    cube = _swap(SOLVED, corner, corner[1:] + corner[:1])
    cube = _swap(cube, edge, edge[::-1])
    response = client.post("/api/solve", json={"cube": cube})
    assert response.json() == {
        "detail": [CubeError.TWISTED_CORNER.value, CubeError.FLIPPED_EDGE.value]
    }


def test_solve_reports_wrong_sticker_counts(client):
    response = client.post("/api/solve", json={"cube": "D" + SOLVED[1:]})
    assert response.json() == {"detail": [CubeError.BAD_COUNTS.value]}


def test_solve_is_post_only(client):
    assert client.get("/api/solve").status_code == 405


def test_scramble_default_is_25_moves(client):
    response = client.get("/api/scramble")
    assert response.status_code == 200
    assert len(response.json()["moves"]) == 25


def test_scramble_moves_make_the_cube(client):
    body = client.get("/api/scramble", params={"num_moves": 20, "seed": 1}).json()
    assert Cube.solved().apply_moves(body["moves"]).to_string() == body["cube"]
    assert body["cube"] == Cube.scramble(20, seed=1).to_string()


def test_scramble_same_seed_same_cube(client):
    first = client.get("/api/scramble", params={"seed": 7}).json()
    second = client.get("/api/scramble", params={"seed": 7}).json()
    assert first == second


def test_scramble_zero_moves_is_solved(client):
    body = client.get("/api/scramble", params={"num_moves": 0}).json()
    assert body == {"cube": SOLVED, "moves": []}


@pytest.mark.parametrize("num_moves, status", [(-1, 422), (100, 200), (101, 422)])
def test_scramble_move_limits(client, num_moves, status):
    response = client.get("/api/scramble", params={"num_moves": num_moves})
    assert response.status_code == status


def test_scramble_then_solve(client):
    params = {"num_moves": 30, "seed": 3}
    cube = client.get("/api/scramble", params=params).json()["cube"]
    moves = client.post("/api/solve", json={"cube": cube}).json()["moves"]
    assert Cube.from_string(cube).apply_moves(moves).is_solved()


def test_apply_one_move(client):
    response = client.post("/api/apply", json={"cube": SOLVED, "moves": ["R"]})
    assert response.status_code == 200
    assert response.json() == {"cube": Cube.solved().apply_move("R").to_string()}


def test_apply_no_moves_keeps_the_cube(client):
    response = client.post("/api/apply", json={"cube": SOLVED, "moves": []})
    assert response.json() == {"cube": SOLVED}


def test_apply_move_then_inverse_returns(client):
    body = {"cube": SOLVED, "moves": ["R", "U", "R'", "U'", "U", "R", "U'", "R'"]}
    assert client.post("/api/apply", json=body).json() == {"cube": SOLVED}


def test_apply_solution_solves_the_cube(client):
    cube = Cube.scramble(30, seed=5).to_string()
    moves = client.post("/api/solve", json={"cube": cube}).json()["moves"]
    response = client.post("/api/apply", json={"cube": cube, "moves": moves})
    assert response.json() == {"cube": SOLVED}


def test_apply_one_step_at_a_time_matches_all_at_once(client):
    cube = Cube.scramble(20, seed=2).to_string()
    moves = ["F", "R2", "U'", "B", "L2"]
    stepped = cube
    for move in moves:
        body = {"cube": stepped, "moves": [move]}
        stepped = client.post("/api/apply", json=body).json()["cube"]
    whole = client.post("/api/apply", json={"cube": cube, "moves": moves}).json()
    assert whole == {"cube": stepped}


@pytest.mark.parametrize("moves", [["X"], ["R", "r"], ["R3"], [""]])
def test_apply_rejects_unknown_move(client, moves):
    response = client.post("/api/apply", json={"cube": SOLVED, "moves": moves})
    assert response.status_code == 422


def test_apply_names_the_unknown_move(client):
    response = client.post("/api/apply", json={"cube": SOLVED, "moves": ["R", "X"]})
    assert response.json() == {"detail": ["Unknown move. Got 'X'."]}


@pytest.mark.parametrize(
    "body",
    [{"cube": SOLVED}, {"moves": ["R"]}, {"cube": "X", "moves": ["R"]}],
)
def test_apply_rejects_bad_request(client, body):
    assert client.post("/api/apply", json=body).status_code == 422


def test_apply_is_post_only(client):
    assert client.get("/api/apply").status_code == 405
