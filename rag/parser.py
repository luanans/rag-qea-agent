import logging
import re
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

SECTION_ALIASES: dict[str, str] = {
    "abstract": "abstract",
    "1 introduction": "introduction",
    "introduction": "introduction",
    "2 background": "related_work",
    "related work": "related_work",
    "background": "related_work",
    "3 model architecture": "methodology",
    "model architecture": "methodology",
    "method": "methodology",
    "methodology": "methodology",
    "approach": "methodology",
    "4 training": "experiments",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "training": "experiments",
    "5 results": "results",
    "results": "results",
    "6 conclusion": "conclusion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "discussion": "discussion",
}


class ParsedSection(NamedTuple):
    name: str
    raw_name: str
    text: str
    page_start: int


class ParsedPaper(NamedTuple):
    paper_id: str
    title: str
    full_text: str
    sections: list[ParsedSection]


def _normalize_section_name(raw: str) -> str | None:
    cleaned = raw.strip().lower()
    cleaned_no_num = re.sub(r"^\d+\.?\s+", "", cleaned)
    return SECTION_ALIASES.get(cleaned) or SECTION_ALIASES.get(cleaned_no_num)


def parse_pdf(pdf_path: Path, paper_id: str, title: str) -> ParsedPaper:
    from docling.document_converter import DocumentConverter

    logger.info("Parsing PDF with Docling: %s", pdf_path)

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    full_text = doc.export_to_markdown()
    sections = _extract_sections(doc)

    logger.info(
        "Parsed '%s': %d chars, %d sections detected",
        paper_id,
        len(full_text),
        len(sections),
    )

    return ParsedPaper(
        paper_id=paper_id,
        title=title,
        full_text=full_text,
        sections=sections,
    )


def _extract_sections(doc) -> list[ParsedSection]:
    from docling_core.types.doc import DocItemLabel

    BODY_LABELS = {
        DocItemLabel.TEXT,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.FORMULA,
    }

    sections: list[ParsedSection] = []
    current_name: str | None = None
    current_raw: str = ""
    current_lines: list[str] = []
    current_page: int = 0

    for item, _ in doc.iterate_items():
        text = getattr(item, "text", "") or ""

        if item.label == DocItemLabel.SECTION_HEADER:
            normalized = _normalize_section_name(text)
            if normalized:
                if current_name and current_lines:
                    sections.append(
                        ParsedSection(
                            name=current_name,
                            raw_name=current_raw,
                            text="\n".join(current_lines).strip(),
                            page_start=current_page,
                        )
                    )
                prov = getattr(item, "prov", None)
                current_name = normalized
                current_raw = text.strip()
                current_lines = []
                current_page = prov[0].page_no if prov else 0

        elif item.label in BODY_LABELS and current_name and text:
            current_lines.append(text)

    if current_name and current_lines:
        sections.append(
            ParsedSection(
                name=current_name,
                raw_name=current_raw,
                text="\n".join(current_lines).strip(),
                page_start=current_page,
            )
        )

    return sections


class SectionStore:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    def add_paper(self, parsed: ParsedPaper) -> None:
        self._store[parsed.paper_id] = {
            section.name: section.text for section in parsed.sections
        }
        logger.debug(
            "SectionStore: added '%s' with sections: %s",
            parsed.paper_id,
            list(self._store[parsed.paper_id].keys()),
        )

    def get_section(self, paper_id: str, section: str) -> str | None:
        paper = self._store.get(paper_id)
        if paper is None:
            return None
        return paper.get(section)

    def list_sections(self, paper_id: str) -> list[str]:
        return list(self._store.get(paper_id, {}).keys())
