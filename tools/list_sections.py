import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from tools.base import BaseTool

if TYPE_CHECKING:
    from rag import SectionStore

logger = logging.getLogger(__name__)

PaperId = Literal["attention", "bert", "rag"]


class ListSectionsInput(BaseModel):
    paper_id: PaperId | None = Field(
        default=None,
        description=(
            "Paper to list sections for. "
            "'attention' = Attention Is All You Need, "
            "'bert' = BERT, "
            "'rag' = Retrieval-Augmented Generation. "
            "Omit to list sections for all papers."
        ),
    )


class PaperSectionList(BaseModel):
    paper_id: str
    sections: list[str]


class ListSectionsOutput(BaseModel):
    papers: list[PaperSectionList]


class ListSectionsTool(BaseTool[ListSectionsInput, ListSectionsOutput]):
    name: str = "list_sections"
    description: str = (
        "Lists the sections available in one or all papers. "
        "Always call this before extract_section when you don't know the exact section name."
    )
    input_model = ListSectionsInput
    output_model = ListSectionsOutput

    def __init__(self, section_store: "SectionStore") -> None:
        self._section_store = section_store

    def _execute(self, payload: ListSectionsInput) -> ListSectionsOutput:
        paper_ids: list[str] = (
            [payload.paper_id] if payload.paper_id else ["attention", "bert", "rag"]
        )

        papers = [
            PaperSectionList(
                paper_id=pid,
                sections=self._section_store.list_sections(pid),
            )
            for pid in paper_ids
        ]

        logger.info(
            "list_sections: paper_id=%s → %s",
            payload.paper_id or "all",
            {p.paper_id: p.sections for p in papers},
        )

        return ListSectionsOutput(papers=papers)
