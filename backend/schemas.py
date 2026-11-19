from pydantic import BaseModel, Field


class CubeRequest(BaseModel):
    cube: str = Field(
        pattern=r"^[URFDLB]{54}$",
        examples=["UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"],
    )


class SolveRequest(CubeRequest):
    pass


class ApplyRequest(CubeRequest):
    moves: list[str] = Field(examples=[["U", "F", "U'", "F'"]])


class SolveResponse(BaseModel):
    moves: list[str]
    length: int


class ScrambleResponse(BaseModel):
    cube: str
    moves: list[str]


class ApplyResponse(BaseModel):
    cube: str
