from fastapi import APIRouter, HTTPException, Query

from backend.schemas import (
    ApplyRequest,
    ApplyResponse,
    ScrambleResponse,
    SolveRequest,
    SolveResponse,
)
from cube.solver import solve
from cube.state import Cube, random_moves
from cube.validation import find_errors

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"health": "ok"}


@router.post("/solve")
def solve_cube(request: SolveRequest) -> SolveResponse:
    cube = Cube.from_string(request.cube)
    errors = find_errors(cube)

    if errors:
        raise HTTPException(status_code=422, detail=[error.value for error in errors])

    moves = solve(cube)
    return SolveResponse(moves=moves, length=len(moves))


@router.post("/apply")
def apply_moves(request: ApplyRequest) -> ApplyResponse:
    try:
        cube = Cube.from_string(request.cube).apply_moves(request.moves)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=[str(error)]) from error

    return ApplyResponse(cube=cube.to_string())


@router.get("/scramble")
def scramble_cube(
    num_moves: int = Query(25, ge=0, le=100), seed: int | None = None
) -> ScrambleResponse:

    moves = random_moves(num_moves, seed)
    cube = Cube.solved().apply_moves(moves)

    return ScrambleResponse(cube=cube.to_string(), moves=moves)
