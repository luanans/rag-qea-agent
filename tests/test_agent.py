from unittest.mock import MagicMock, patch

import pytest

from agent.qa_agent import QAAgent


def _make_function_call_response(tool_name: str, args: dict):
    fc = MagicMock()
    fc.name = tool_name
    fc.args = args

    part = MagicMock()
    part.function_call = fc
    part.text = None

    candidate = MagicMock()
    candidate.content.parts = [part]

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_text_response(text: str):
    part = MagicMock()
    part.function_call = None
    part.text = text

    candidate = MagicMock()
    candidate.content.parts = [part]

    response = MagicMock()
    response.candidates = [candidate]
    return response


class TestQAAgent:
    @pytest.fixture
    def agent(self, registry_with_mocks):
        with patch("agent.qa_agent.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            agent = QAAgent(
                registry=registry_with_mocks,
                api_key="fake-key",
                model="gemini-2.5-flash",
                max_iterations=5,
            )
            agent._mock_client = mock_client
            yield agent

    def test_direct_text_response(self, agent):
        """Gemini replies directly without calling any tool."""
        agent._mock_client.models.generate_content.return_value = _make_text_response(
            "The Transformer uses self-attention."
        )

        result = agent.answer("What is the Transformer?")

        assert result == "The Transformer uses self-attention."
        assert agent._mock_client.models.generate_content.call_count == 1

    def test_single_tool_call_then_answer(self, agent):
        """Gemini calls search_documents once, then produces the final answer."""
        agent._mock_client.models.generate_content.side_effect = [
            _make_function_call_response(
                "search_documents",
                {"query": "attention mechanism", "top_k": 5},
            ),
            _make_text_response("The Transformer relies on multi-head self-attention."),
        ]

        result = agent.answer("How does the attention mechanism work?")

        assert "attention" in result.lower() or "Transformer" in result
        assert agent._mock_client.models.generate_content.call_count == 2

    def test_two_tool_calls_then_answer(self, agent):
        """Agent calls two tools before producing the final answer."""
        agent._mock_client.models.generate_content.side_effect = [
            _make_function_call_response(
                "search_documents",
                {"query": "BERT pretraining", "top_k": 5},
            ),
            _make_function_call_response(
                "extract_section",
                {"paper_id": "bert", "section": "abstract"},
            ),
            _make_text_response("BERT uses bidirectional training of Transformers."),
        ]

        result = agent.answer("How is BERT pre-trained?")

        assert agent._mock_client.models.generate_content.call_count == 3
        assert result == "BERT uses bidirectional training of Transformers."

    def test_max_iterations_reached(self, agent):
        """Agent stops after max_iterations without producing a final answer."""
        agent._max_iterations = 2
        agent._mock_client.models.generate_content.return_value = (
            _make_function_call_response("search_documents", {"query": "test"})
        )

        result = agent.answer("Question that causes an infinite loop")

        assert "limit" in result.lower() or "maximum" in result.lower()
        assert agent._mock_client.models.generate_content.call_count == 2

    def test_unknown_tool_call_returns_gracefully(self, agent):
        """Gemini calls a nonexistent tool — agent should handle it gracefully."""
        agent._mock_client.models.generate_content.side_effect = [
            _make_function_call_response("nonexistent_tool", {"foo": "bar"}),
            _make_text_response("I could not find information."),
        ]

        result = agent.answer("Any question")

        assert result == "I could not find information."

    def test_registry_schemas_sent_to_gemini(self, agent):
        """Tool schemas from the registry are passed to Gemini on every call."""
        agent._mock_client.models.generate_content.return_value = _make_text_response(
            "ok"
        )

        agent.answer("Any question")

        call_kwargs = agent._mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config")
        if config is None and len(call_kwargs.args) > 2:
            config = call_kwargs.args[2]

        assert config is not None
        assert config.tools is not None
