import logging
from typing import Literal

from pydantic import BaseModel, Field

from tools.base import BaseTool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag import SectionStore

logger = logging.getLogger(__name__)

PaperId = Literal["attention", "bert", "rag"]
SectionName = Literal[
    "abstract",
    "introduction",
    "conclusion",
    "related_work",
    "methodology",
    "experiments",
    "results",
    "discussion",
]

PAPER_TITLES: dict[str, str] = {
    "attention": "Attention Is All You Need",
    "bert": "BERT: Pre-training of Deep Bidirectional Transformers",
    "rag": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
}


class ExtractSectionInput(BaseModel):
    paper_id: PaperId = Field(
        ...,
        description=(
            "Identificador do paper: "
            "'attention' = Attention Is All You Need, "
            "'bert' = BERT, "
            "'rag' = Retrieval-Augmented Generation."
        ),
    )
    section: SectionName = Field(
        ...,
        description=(
            "Seção a extrair. Valores válidos: abstract, introduction, conclusion, "
            "related_work, methodology, experiments, results, discussion."
        ),
    )


class ExtractSectionOutput(BaseModel):
    paper_id: str
    title: str
    section: str
    text: str
    char_count: int


class ExtractSectionTool(BaseTool[ExtractSectionInput, ExtractSectionOutput]):
    name: str = "extract_section"
    description: str = (
        "Extrai o texto completo de uma seção específica de um paper "
        "(ex: abstract, introduction, conclusion). "
        "Use quando precisar do conteúdo integral de uma seção para embasar a resposta."
    )
    input_model = ExtractSectionInput
    output_model = ExtractSectionOutput

    def __init__(self, section_store: "SectionStore") -> None:
        self._section_store = section_store

    def _execute(self, payload: ExtractSectionInput) -> ExtractSectionOutput:
        text = self._section_store.get_section(
            paper_id=payload.paper_id,
            section=payload.section,
        )

        if text is None:
            raise ValueError(
                f"Section '{payload.section}' not found in paper '{payload.paper_id}'. "
                f"Try search_documents instead, or use a different section name."
            )

        title = PAPER_TITLES[payload.paper_id]

        logger.info(
            "extract_section: paper='%s' section='%s' → %d chars",
            payload.paper_id,
            payload.section,
            len(text),
        )

        return ExtractSectionOutput(
            paper_id=payload.paper_id,
            title=title,
            section=payload.section,
            text=text,
            char_count=len(text),
        )
