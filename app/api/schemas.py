from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the multi-head attention mechanism in the Transformer?"
            }
        }
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Question about the papers. "
            "Supported papers: *Attention Is All You Need*, *BERT*, and *RAG*.\n\n"
            "Accepted in any language supported by Gemini — English, Portuguese, Spanish, "
            "French, German, Japanese, Chinese, Arabic, Hindi, and "
            "[100+ others](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models?hl=pt-br#expandable-1). "
            "The agent replies in the same language as the question."
        ),
        examples=[
            "What is the multi-head attention mechanism in the Transformer?",
            "Qual é a diferença entre BERT e o modelo RAG?",
        ],
    )


class AskResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the multi-head attention mechanism in the Transformer?",
                "answer": (
                    "Multi-head attention allows the model to jointly attend to information "
                    "from different representation subspaces at different positions. "
                    "Instead of performing a single attention function with d_model-dimensional "
                    "keys, values and queries, the authors project these h times with different "
                    "learned linear projections to d_k, d_k and d_v dimensions respectively "
                    "(Attention Is All You Need, Section 3.2.2)."
                ),
            }
        }
    )

    question: str = Field(..., description="The original question sent by the user.")
    answer: str = Field(
        ...,
        description=(
            "Answer synthesized by the agent, grounded in the papers' content. "
            "Includes citations to the source paper and section."
        ),
    )
