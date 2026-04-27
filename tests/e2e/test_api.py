import pytest


pytestmark = pytest.mark.e2e


class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_returns_docs_url(self, client):
        r = client.get("/")
        assert "docs" in r.json()


class TestAskValidation:
    def test_empty_question_rejected(self, client):
        r = client.post("/ask", json={"question": ""})
        assert r.status_code == 422

    def test_missing_question_field_rejected(self, client):
        r = client.post("/ask", json={})
        assert r.status_code == 422

    def test_question_exceeding_max_length_rejected(self, client):
        r = client.post("/ask", json={"question": "x" * 1001})
        assert r.status_code == 422

    def test_invalid_body_type_rejected(self, client):
        r = client.post(
            "/ask", content="not json", headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 422


class TestAskE2E:
    def test_returns_200_with_answer(self, client):
        r = client.post(
            "/ask", json={"question": "What is the Transformer architecture?"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["question"] == "What is the Transformer architecture?"
        assert isinstance(body["answer"], str)
        assert len(body["answer"]) > 0

    def test_response_echoes_question(self, client):
        question = "What is BERT?"
        r = client.post("/ask", json={"question": question})
        assert r.status_code == 200
        assert r.json()["question"] == question

    def test_answer_is_grounded_in_papers(self, client):
        r = client.post(
            "/ask",
            json={"question": "What is the multi-head attention mechanism?"},
        )
        assert r.status_code == 200
        answer = r.json()["answer"].lower()
        assert any(
            keyword in answer for keyword in ["attention", "transformer", "head"]
        )

    def test_question_about_bert(self, client):
        r = client.post(
            "/ask",
            json={"question": "What pre-training tasks does BERT use?"},
        )
        assert r.status_code == 200
        answer = r.json()["answer"].lower()
        assert any(
            keyword in answer
            for keyword in ["masked", "bert", "pre-train", "language model"]
        )

    def test_question_about_rag(self, client):
        r = client.post(
            "/ask",
            json={"question": "How does RAG combine retrieval with generation?"},
        )
        assert r.status_code == 200
        answer = r.json()["answer"].lower()
        assert any(
            keyword in answer
            for keyword in ["retrieval", "generation", "rag", "knowledge"]
        )

    def test_question_in_portuguese(self, client):
        r = client.post(
            "/ask",
            json={"question": "O que é atenção multi-cabeça no Transformer?"},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["answer"]) > 0

    def test_out_of_scope_question_still_responds(self, client):
        r = client.post(
            "/ask",
            json={"question": "What is the capital of France?"},
        )
        assert r.status_code == 200
        assert isinstance(r.json()["answer"], str)
