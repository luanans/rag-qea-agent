import json

from tools.search_documents import SearchDocumentsTool


class TestSearchDocumentsTool:
    def test_returns_matching_chunks(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        result = tool.run({"query": "attention mechanism"})

        assert result.success is True
        data = json.loads(result.output)
        assert data["total_found"] == 1
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["paper_id"] == "attention"
        assert data["chunks"][0]["score"] == 0.92

    def test_passes_query_to_vector_store(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        tool.run({"query": "multi-head attention"})

        mock_vector_store.query.assert_called_once()
        kwargs = mock_vector_store.query.call_args.kwargs
        assert kwargs.get("query_text") == "multi-head attention"

    def test_paper_id_filter_forwarded(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        tool.run({"query": "BERT embeddings", "paper_id": "bert"})

        kwargs = mock_vector_store.query.call_args.kwargs
        assert kwargs.get("where") == {"paper_id": "bert"}

    def test_no_paper_id_sends_none_filter(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        tool.run({"query": "transformer model"})

        kwargs = mock_vector_store.query.call_args.kwargs
        assert kwargs.get("where") is None

    def test_empty_results(self, mocker):
        store = mocker.MagicMock()
        store.query.return_value = []
        tool = SearchDocumentsTool(vector_store=store)
        result = tool.run({"query": "unknown topic"})

        assert result.success is True
        data = json.loads(result.output)
        assert data["total_found"] == 0
        assert data["chunks"] == []

    def test_top_k_forwarded(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        tool.run({"query": "positional encoding", "top_k": 7})

        kwargs = mock_vector_store.query.call_args.kwargs
        assert kwargs.get("top_k") == 7

    def test_query_too_short_returns_failure(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        result = tool.run({"query": "ab"})

        assert result.success is False

    def test_invalid_paper_id_returns_failure(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        result = tool.run({"query": "attention mechanism", "paper_id": "gpt4"})

        assert result.success is False

    def test_gemini_schema_structure(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        schema = tool.to_gemini_schema()

        assert schema["name"] == "search_documents"
        assert "description" in schema
        props = schema["parameters"].get("properties", {})
        assert "query" in props
        assert "top_k" in props
        assert "paper_id" in props

    def test_chunk_fields_present(self, mock_vector_store):
        tool = SearchDocumentsTool(vector_store=mock_vector_store)
        result = tool.run({"query": "attention mechanism"})

        data = json.loads(result.output)
        chunk = data["chunks"][0]
        for field in ("chunk_id", "paper_id", "title", "section", "text", "score"):
            assert field in chunk
