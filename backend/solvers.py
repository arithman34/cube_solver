import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from cube.solver import solve
from cube.state import Cube

MODEL_PATH = Path(
    os.environ.get(
        "CUBE_MODEL_PATH",
        Path(__file__).parent.parent
        / "training"
        / "checkpoints"
        / "value_iteration_overnight.pt",
    )
)


class SolverUnavailable(Exception): ...


class SolveFailed(Exception): ...


@dataclass
class Solver:
    """solve takes the cube and a time limit in seconds. Solvers with
    uses_time_limit False ignore the limit."""

    name: str
    label: str
    solve: Callable[[Cube, float], list[str]]
    uses_time_limit: bool = False


def kociemba_solver() -> Solver:
    return Solver(
        name="kociemba",
        label="Kociemba (two-phase)",
        solve=solve,
        uses_time_limit=True,
    )


def neural_solver(path: Path, beam_width: int = 300) -> Solver:
    """Beam search guided by the value network saved at path, loaded once.

    Raises SolverUnavailable if torch is not installed or the file is missing.
    The returned solver raises SolveFailed when the search gives up.
    """
    try:
        import torch

        from training.search import beam_search
        from training.train import load_checkpoint
    except ImportError as error:
        raise SolverUnavailable("torch is not installed") from error

    if not path.exists():
        raise SolverUnavailable(f"No checkpoint at {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(path, device)

    def solve_with_beam(cube: Cube, time_limit: float = 0) -> list[str]:
        result = beam_search(model, cube, device, beam_width)
        if not result.solved:
            raise SolveFailed(
                "The neural solver couldn't find a solution. Try Kociemba."
            )
        return result.moves

    return Solver(
        name="neural", label="Neural (value iteration)", solve=solve_with_beam
    )


@cache
def available_solvers() -> dict[str, Solver]:
    """Kociemba is always available. Neural only if its checkpoint loads."""
    solvers = [kociemba_solver()]
    try:
        solvers.append(neural_solver(MODEL_PATH))
    except SolverUnavailable as error:
        print(f"Neural solver unavailable: {error}")

    return {solver.name: solver for solver in solvers}
