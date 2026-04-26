import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import AskRequest, AskResponse
from app.dependencies import get_agent

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    summary="Health check",
    response_description="Service status and docs URL.",
)
def health() -> dict:
    return {"status": "ok", "docs": "/docs"}


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question to the Q&A agent",
    description=(
        "Sends a natural language question to the agent. "
        "The agent uses semantic search (`search_documents`) and full-section retrieval "
        "(`extract_section`) over three ML papers — *Attention Is All You Need*, *BERT*, "
        "and *Retrieval-Augmented Generation* — and synthesizes an answer grounded in "
        "the original texts.\n\n"
        "**Supported topics:** Transformer architecture, multi-head attention, positional "
        "encoding, BERT pre-training, masked language modelling, RAG retrieval pipeline, "
        "knowledge-intensive NLP tasks, and more.\n"
        "**Supported Languages:**  Question accepted in any language supported by Gemini — English, Portuguese, Spanish, French and 100+ others. The agent replies in the same language as the question."
    ),
    response_description="The agent's answer, grounded in the papers' content.",
    responses={
        429: {
            "description": "Gemini API rate limit reached. Wait a moment and retry.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Gemini API rate limit reached. Wait a moment and try again."
                    }
                }
            },
        },
        500: {
            "description": "Internal agent error.",
            "content": {
                "application/json": {
                    "example": {"detail": "Agent error: <error message>"}
                }
            },
        },
    },
)
def ask(
    request: AskRequest,
    agent=Depends(get_agent),
) -> AskResponse:
    logger.info("POST /ask — question='%s'", request.question)

    try:
        answer = agent.answer(request.question)
    except Exception as exc:
        try:
            from google.genai.errors import ClientError as GeminiClientError
        except ImportError:
            GeminiClientError = None  # type: ignore[assignment]
        if (
            GeminiClientError
            and isinstance(exc, GeminiClientError)
            and exc.status_code == 429
        ):
            logger.warning("Gemini rate limit hit: %s", exc)
            raise HTTPException(
                status_code=429,
                detail="Gemini API rate limit reached. Wait a moment and try again.",
            ) from exc
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return AskResponse(question=request.question, answer=answer)
