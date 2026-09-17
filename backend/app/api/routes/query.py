import logging
from typing import Annotated

import httpx
import ollama
from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings
from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService

router = APIRouter()
logger = logging.getLogger(__name__)


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval


def get_generation_service(request: Request) -> GenerationService:
    return request.app.state.generation


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/health")
def health(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="RAG services are not ready")
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    retrieval: Annotated[RetrievalService, Depends(get_retrieval_service)],
    generation: Annotated[GenerationService, Depends(get_generation_service)],
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> QueryResponse:
    try:
        chunks = retrieval.retrieve(payload.question, k=settings.retrieval_k)
        answer = generation.answer(payload.question, chunks)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Ollama request timed out") from exc
    except (ConnectionError, httpx.RequestError, ollama.ResponseError) as exc:
        raise HTTPException(status_code=503, detail="Ollama is unavailable") from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail="Unable to answer the query") from exc
    return QueryResponse(
        answer=answer,
        sources=list(dict.fromkeys(chunk["source"] for chunk in chunks)),
    )
