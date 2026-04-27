import logging
import os
from typing import Callable

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_GEMINI_MODELS = frozenset({"gemini-embedding-001", "text-embedding-004"})


def build_embedding_fn(
    model_name: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    api_key: str | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    if model_name in _GEMINI_MODELS or model_name.startswith("models/"):
        return _build_gemini_fn(model_name, task_type, api_key)
    return _build_st_fn(model_name)


def _build_gemini_fn(
    model_name: str,
    task_type: str,
    api_key: str | None,
) -> Callable[[list[str]], list[list[float]]]:
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY is required for Gemini models.")

    client = genai.Client(api_key=key)
    logger.info("Gemini embedding fn: model=%s task_type=%s", model_name, task_type)

    def embed(texts: list[str]) -> list[list[float]]:
        batch_size = 100
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.models.embed_content(
                model=model_name,
                contents=batch,
                config={"task_type": task_type},
            )
            results.extend(list(e.values) for e in resp.embeddings)
        return results

    return embed


def _build_st_fn(model_name: str) -> Callable[[list[str]], list[list[float]]]:
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded.")

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        return vectors.tolist()

    return embed
