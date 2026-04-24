import logging
import pickle
from pathlib import Path

from app.config import configure_logging, get_settings
from rag.chunker import chunk_paper
from rag.embeddings import build_embedding_fn
from rag.loader import PAPERS, download_papers
from rag.parser import SectionStore, parse_pdf
from rag.vector_store import VectorStore

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()

    logger.info("=== RAG Ingest Pipeline ===")

    logger.info("Step 1/4: Downloading papers...")
    pdf_paths = download_papers(output_dir="./data/papers")

    logger.info("Step 2/4: Parsing PDFs...")
    section_store = SectionStore()
    parsed_papers = []

    for paper_id, pdf_path in pdf_paths.items():
        title = PAPERS[paper_id]["title"]
        parsed = parse_pdf(pdf_path=pdf_path, paper_id=paper_id, title=title)
        section_store.add_paper(parsed)
        parsed_papers.append(parsed)
        logger.info(
            "  '%s': %d sections, %d chars",
            paper_id,
            len(parsed.sections),
            len(parsed.full_text),
        )

    logger.info("Step 3/4: Persisting SectionStore...")
    cache_dir = Path("./persist")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "sections.pkl"

    with open(cache_path, "wb") as f:
        pickle.dump(section_store, f)
    logger.info("SectionStore saved to %s", cache_path)

    logger.info("Step 4/4: Building vector store...")
    doc_embed = build_embedding_fn(
        settings.embedding_model,
        task_type="RETRIEVAL_DOCUMENT",
        api_key=settings.gemini_api_key,
    )
    vector_store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding_fn=doc_embed,
    )

    if vector_store.count > 0:
        logger.info(
            "Vector store already has %d chunks. Skipping re-ingestion. "
            "Delete '%s' to force re-ingest.",
            vector_store.count,
            settings.chroma_persist_dir,
        )
    else:
        total_chunks = 0
        for parsed in parsed_papers:
            chunks = chunk_paper(
                paper=parsed,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            vector_store.add_chunks(chunks)
            total_chunks += len(chunks)
            logger.info("  '%s': %d chunks added.", parsed.paper_id, len(chunks))

        logger.info("Vector store built: %d total chunks.", total_chunks)

    logger.info("=== Ingest complete. Run: uvicorn app.main:app ===")


if __name__ == "__main__":
    main()
