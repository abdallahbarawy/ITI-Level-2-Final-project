from typing import Annotated

from pydantic import BaseModel, StringConstraints


class QueryRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=1000)
    ]


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
