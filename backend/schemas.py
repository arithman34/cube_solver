from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    cube: str = Field(
        pattern=r"^[URFDLB]{54}$",
        examples=["UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"],
    )


class SolveResponse(BaseModel):
    moves: list[str]
    length: int


class ScrambleResponse(BaseModel):
    cube: str
    moves: list[str]
