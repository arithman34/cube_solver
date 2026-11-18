from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routes import router
from cube.solver.tables import get_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_tables()
    yield


app = FastAPI(title="Cube Solver", lifespan=lifespan)
app.include_router(router)
