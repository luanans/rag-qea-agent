import pytest

from rag.parser import ParsedPaper, ParsedSection, SectionStore
from tools.registry import ToolRegistry


ATTENTION_ABSTRACT = (
    "The dominant sequence transduction models are based on complex recurrent or convolutional "
    "neural networks. We propose the Transformer, a model architecture eschewing recurrence "
    "and instead relying entirely on an attention mechanism to draw global dependencies between "
    "input and output."
)

ATTENTION_INTRODUCTION = (
    "Recurrent neural networks, long short-term memory and gated recurrent neural networks "
    "in particular, have been firmly established as state of the art approaches in sequence "
    "modeling and transduction problems such as language modeling and machine translation. "
    "The Transformer allows for significantly more parallelization."
)

BERT_ABSTRACT = (
    "We introduce a new language representation model called BERT, which stands for "
    "Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train "
    "deep bidirectional representations from unlabeled text by jointly conditioning on both "
    "left and right context in all layers."
)

RAG_ABSTRACT = (
    "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG), "
    "models which combine pre-trained parametric and non-parametric memory for language generation. "
    "We endow the models with access to a dense vector index of Wikipedia."
)


@pytest.fixture
def section_store() -> SectionStore:
    """SectionStore populated with fixture data (no disk I/O)."""
    store = SectionStore()

    attention = ParsedPaper(
        paper_id="attention",
        title="Attention Is All You Need",
        full_text=ATTENTION_ABSTRACT + "\n" + ATTENTION_INTRODUCTION,
        sections=[
            ParsedSection("abstract", "Abstract", ATTENTION_ABSTRACT, 0),
            ParsedSection("introduction", "1 Introduction", ATTENTION_INTRODUCTION, 1),
        ],
    )

    bert = ParsedPaper(
        paper_id="bert",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        full_text=BERT_ABSTRACT,
        sections=[
            ParsedSection("abstract", "Abstract", BERT_ABSTRACT, 0),
        ],
    )

    rag = ParsedPaper(
        paper_id="rag",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        full_text=RAG_ABSTRACT,
        sections=[
            ParsedSection("abstract", "Abstract", RAG_ABSTRACT, 0),
        ],
    )

    store.add_paper(attention)
    store.add_paper(bert)
    store.add_paper(rag)

    return store


@pytest.fixture
def mock_vector_store(mocker):
    """VectorStore mock that returns fixture chunks."""
    store = mocker.MagicMock()
    store.query.return_value = [
        {
            "id": "attention__abstract__0",
            "document": ATTENTION_ABSTRACT,
            "metadata": {
                "paper_id": "attention",
                "title": "Attention Is All You Need",
                "section": "abstract",
                "chunk_index": 0,
                "source": "attention/abstract",
            },
            "score": 0.92,
        }
    ]
    return store


@pytest.fixture
def registry_with_mocks(mock_vector_store, section_store):
    """Full ToolRegistry using mocked dependencies."""
    from tools.extract_section import ExtractSectionTool
    from tools.search_documents import SearchDocumentsTool

    registry = ToolRegistry()
    registry.register(SearchDocumentsTool(vector_store=mock_vector_store))
    registry.register(ExtractSectionTool(section_store=section_store))
    return registry
