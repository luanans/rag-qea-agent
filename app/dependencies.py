import logging
from functools import lru_cache


from rag.embeddings import build_embedding_fn
from rag.parser import SectionStore
from agent.qa_agent import QAAgent
from app.config import get_settings
from rag.vector_store import VectorStore
from tools.extract_section import ExtractSectionTool
from tools.list_sections import ListSectionsTool
from tools.registry import ToolRegistry
from tools.search_documents import SearchDocumentsTool

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    doc_embed = build_embedding_fn(
        settings.embedding_model,
        task_type="RETRIEVAL_DOCUMENT",
        api_key=settings.gemini_api_key,
    )
    query_embed = build_embedding_fn(
        settings.embedding_model,
        task_type="RETRIEVAL_QUERY",
        api_key=settings.gemini_api_key,
    )
    store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding_fn=doc_embed,
        query_embedding_fn=query_embed,
    )
    logger.info("VectorStore singleton created. count=%d", store.count)
    return store


@lru_cache(maxsize=1)
def get_section_store() -> SectionStore:
    import pickle
    from pathlib import Path

    cache_path = Path("./persist/sections.pkl")

    if cache_path.exists():
        logger.info("Loading SectionStore from cache: %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.warning(
        "SectionStore cache not found at %s. "
        "Run ingest.py first to populate the vector store and section cache.",
        cache_path,
    )
    return SectionStore()


@lru_cache(maxsize=1)
def get_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SearchDocumentsTool(vector_store=get_vector_store()))
    registry.register(ListSectionsTool(section_store=get_section_store()))
    registry.register(ExtractSectionTool(section_store=get_section_store()))
    logger.info("ToolRegistry built with tools: %s", registry.tool_names)
    return registry


@lru_cache(maxsize=1)
def get_agent() -> QAAgent:
    settings = get_settings()
    agent = QAAgent(
        registry=get_registry(),
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_iterations=settings.agent_max_iterations,
        temperature=settings.gemini_temperature,
        top_p=settings.gemini_top_p,
    )
    return agent
