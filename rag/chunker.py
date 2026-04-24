import logging
from typing import Any

from rag.parser import ParsedPaper

logger = logging.getLogger(__name__)


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def chunk_paper(
    paper: ParsedPaper,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []

    if paper.sections:
        for sec_idx, section in enumerate(paper.sections):
            section_chunks = _split_into_chunks(section.text, chunk_size, overlap)
            for i, text in enumerate(section_chunks):
                chunk_id = f"{paper.paper_id}__{section.name}_{sec_idx}__{i}"
                all_chunks.append(
                    {
                        "id": chunk_id,
                        "document": text,
                        "metadata": {
                            "paper_id": paper.paper_id,
                            "title": paper.title,
                            "section": section.name,
                            "chunk_index": i,
                            "source": f"{paper.paper_id}/{section.name}",
                        },
                    }
                )
    else:
        logger.warning(
            "No sections detected for '%s'. Falling back to full-text chunking.",
            paper.paper_id,
        )
        full_chunks = _split_into_chunks(paper.full_text, chunk_size, overlap)
        for i, text in enumerate(full_chunks):
            all_chunks.append(
                {
                    "id": f"{paper.paper_id}__full__{i}",
                    "document": text,
                    "metadata": {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "section": "unknown",
                        "chunk_index": i,
                        "source": f"{paper.paper_id}/full",
                    },
                }
            )

    logger.info(
        "Chunked '%s': %d sections → %d chunks (size=%d, overlap=%d)",
        paper.paper_id,
        len(paper.sections),
        len(all_chunks),
        chunk_size,
        overlap,
    )

    return all_chunks
