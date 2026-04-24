import logging
from typing import Literal

from pydantic import BaseModel, Field

from tools.base import BaseTool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag import VectorStore
logger = logging.getLogger(__name__)

PaperId = Literal["attention", "bert", "rag"]


class ChunkResult(BaseModel):
    """A chunk returned by a semantic search query."""

    chunk_id: str
    paper_id: str
    title: str
    section: str
    text: str
    score: float


class SearchDocumentsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural language query to find relevant passages in the papers. "
            "Be specific: prefer 'multi-head attention mechanism' over 'attention'."
        ),
        min_length=3,
    )
    top_k: int = Field(
        default=5,
        description="Number of chunks to return. Use 3-5 for focused questions, 7-10 for a broader overview.",
        ge=1,
        le=10,
    )
    paper_id: PaperId | None = Field(
        default=None,
        description=(
            "Restricts the search to a specific paper. "
            "'attention' = Attention Is All You Need, "
            "'bert' = BERT, "
            "'rag' = Retrieval-Augmented Generation. "
            "Omit to search across all papers."
        ),
    )


class SearchDocumentsOutput(BaseModel):
    chunks: list[ChunkResult]
    total_found: int


class SearchDocumentsTool(BaseTool[SearchDocumentsInput, SearchDocumentsOutput]):
    """
    Semantic search over the vector store to retrieve the most relevant chunks for a query.
    Supports filtering by paper and controlling the number of results.
    """

    name: str = "search_documents"
    description: str = (
        "Searches for relevant passages in the scientific papers using semantic similarity. "
        "Use this tool to find specific information about concepts, mechanisms, "
        "architectures, or results mentioned in the papers."
    )
    input_model = SearchDocumentsInput
    output_model = SearchDocumentsOutput

    def __init__(self, vector_store: "VectorStore") -> None:  # type: ignore[name-defined]
        self._vector_store = vector_store

    def _execute(self, payload: SearchDocumentsInput) -> SearchDocumentsOutput:
        where_filter = {"paper_id": payload.paper_id} if payload.paper_id else None

        raw_results = self._vector_store.query(
            query_text=payload.query,
            top_k=payload.top_k,
            where=where_filter,
        )

        chunks = [
            ChunkResult(
                chunk_id=r["id"],
                paper_id=r["metadata"]["paper_id"],
                title=r["metadata"]["title"],
                section=r["metadata"]["section"],
                text=r["document"],
                score=r["score"],
            )
            for r in raw_results
        ]

        logger.info(
            "search_documents: query='%s' paper_id=%s → %d chunks found",
            payload.query,
            payload.paper_id,
            len(chunks),
        )

        return SearchDocumentsOutput(chunks=chunks, total_found=len(chunks))
