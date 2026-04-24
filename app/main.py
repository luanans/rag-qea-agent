from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import router
from app.config import configure_logging, get_settings
from app.dependencies import get_agent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()

    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting RAG Q&A Agent — model=%s", settings.gemini_model)

    get_agent()

    yield

    logger.info("Shutting down RAG Q&A Agent.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Q&A Agent",
        description=(
            "Agente de perguntas e respostas sobre artigos científicos de ML. "
            "Powered by Gemini 2.0 Flash + ChromaDB + sentence-transformers."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(router, tags=["Q&A"])

    return app


app = create_app()
