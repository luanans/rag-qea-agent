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
            "Q&A agent over scientific ML papers, powered by **Gemini 2.5 Flash** "
            "with native function calling, **ChromaDB** for vector search, and "
            "**gemini-embedding-001** for semantic embeddings.\n\n"
            "## Papers covered\n"
            "- **Attention Is All You Need** — Vaswani et al., 2017 (`attention`)\n"
            "- **BERT: Pre-training of Deep Bidirectional Transformers** — Devlin et al., 2018 (`bert`)\n"
            "- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al., 2020 (`rag`)\n\n"
            "## Rate limits\n"
            "Requests are subject to the Gemini API quota. If you receive a `429`, wait a few seconds and retry."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(router, tags=["Q&A"])

    return app


app = create_app()
