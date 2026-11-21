import time

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import (
    ApplyRequest,
    ApplyResponse,
    MethodInfo,
    ScrambleResponse,
    SolveRequest,
    SolveResponse,
)
from backend.solvers import SolveFailed, available_solvers
from cube.state import Cube, random_moves
from cube.validation import find_errors

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"health": "ok"}


@router.post("/solve")
def solve_cube(request: SolveRequest) -> SolveResponse:
    solvers = available_solvers()
    if request.method not in solvers:
        raise HTTPException(
            status_code=422,
            detail=[
                f"Unknown method {request.method!r}. Choose from: {', '.join(solvers)}"
            ],
        )

    cube = Cube.from_string(request.cube)
    errors = find_errors(cube)

    if errors:
        raise HTTPException(status_code=422, detail=[error.value for error in errors])

    solver = solvers[request.method]
    start = time.perf_counter()
    try:
        moves = solver.solve(cube)
    except SolveFailed as error:
        raise HTTPException(status_code=422, detail=[str(error)]) from error
    time_ms = (time.perf_counter() - start) * 1000
    return SolveResponse(
        moves=moves, length=len(moves), method=request.method, time_ms=round(time_ms, 2)
    )


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


@router.get("/methods")
def methods() -> list[MethodInfo]:
    return [
        MethodInfo(name=solver.name, label=solver.label)
        for solver in available_solvers().values()
    ]
