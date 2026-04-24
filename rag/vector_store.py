import logging
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_fn: Any,
        query_embedding_fn: Any | None = None,
    ) -> None:
        import os
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=os.path.abspath(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedding_fn = embedding_fn
        self._query_embedding_fn = query_embedding_fn or embedding_fn
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore initialized: collection='%s' persist_dir='%s'",
            collection_name,
            persist_dir,
        )

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            logger.warning("add_chunks called with empty list.")
            return

        ids = [c["id"] for c in chunks]
        documents = [c["document"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        embeddings = self._embedding_fn(documents)

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Added %d chunks to vector store.", len(chunks))

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self._query_embedding_fn([query_text])[0]

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output: list[dict[str, Any]] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            output.append(
                {
                    "id": chunk_id,
                    "document": doc,
                    "metadata": meta,
                    "score": round(1 - dist, 4),
                }
            )

        return output

    @property
    def count(self) -> int:
        return self._collection.count()
