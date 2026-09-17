from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api.routes.query import router
from app.core.config import Settings, get_settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        config = settings.load_store_config()
        generation = GenerationService(settings, config)
        try:
            await run_in_threadpool(generation.check_connection)
            app.state.retrieval = await run_in_threadpool(RetrievalService, settings, config)
            app.state.generation = generation
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            generation.close()
            app.state.retrieval = None
            app.state.generation = None

    app = FastAPI(title="Football RAG Assistant", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
