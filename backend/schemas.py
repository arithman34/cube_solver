from pydantic import BaseModel, Field

MAX_TIME_LIMIT = 10.0


class CubeRequest(BaseModel):
    cube: str = Field(
        pattern=r"^[URFDLB]{54}$",
        examples=["UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"],
    )


class SolveRequest(CubeRequest):
    method: str = "kociemba"
    time_limit: float = Field(1.0, ge=0, le=MAX_TIME_LIMIT)


class ApplyRequest(CubeRequest):
    moves: list[str] = Field(examples=[["U", "F", "U'", "F'"]])


class SolveResponse(BaseModel):
    moves: list[str]
    length: int
    method: str
    time_ms: float


class ScrambleResponse(BaseModel):
    cube: str
    moves: list[str]


class ApplyResponse(BaseModel):
    cube: str


class MethodInfo(BaseModel):
    name: str
    label: str
    uses_time_limit: bool
