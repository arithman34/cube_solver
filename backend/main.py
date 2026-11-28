from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes import router
from backend.solvers import available_solvers
from cube.kociemba.tables import get_tables

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_tables()
    available_solvers()
    yield


app = FastAPI(title="Cube Solver", lifespan=lifespan)
app.include_router(router, prefix="/api")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")
