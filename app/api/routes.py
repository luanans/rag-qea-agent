import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import AskRequest, AskResponse
from app.dependencies import get_agent

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", summary="Health check")
def health() -> dict:
    return {"status": "ok", "docs": "/docs"}


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Faz uma pergunta ao agente Q&A",
    description=(
        "Envia uma pergunta em linguagem natural ao agente. "
        "O agente busca nos papers Attention Is All You Need, BERT e RAG "
        "e retorna uma resposta embasada nos textos originais."
    ),
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
