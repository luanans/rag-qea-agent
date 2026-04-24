from tools.extract_section import ExtractSectionTool


class TestExtractSectionTool:
    def test_extract_existing_section(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)
        result = tool.run({"paper_id": "attention", "section": "abstract"})

        assert result.success is True
        assert result.data is not None
        assert result.data.paper_id == "attention"
        assert result.data.section == "abstract"
        assert "Transformer" in result.data.text
        assert result.data.char_count > 0

    def test_extract_returns_correct_title(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)
        result = tool.run({"paper_id": "bert", "section": "abstract"})

        assert result.success is True
        assert "BERT" in result.data.title

    def test_extract_nonexistent_section_returns_failure(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)
        # 'results' não foi adicionado no fixture
        result = tool.run({"paper_id": "attention", "section": "results"})

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_invalid_paper_id_returns_failure(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)
        result = tool.run({"paper_id": "gpt4", "section": "abstract"})

        assert result.success is False
        assert "validation" in result.error.lower()

    def test_invalid_section_name_returns_failure(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)
        result = tool.run({"paper_id": "attention", "section": "random_section"})

        assert result.success is False

    def test_missing_required_fields_returns_failure(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)
        result = tool.run({"paper_id": "rag"})  # falta 'section'

        assert result.success is False

    def test_char_count_matches_text_length(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)
        result = tool.run({"paper_id": "rag", "section": "abstract"})

        assert result.success is True
        assert result.data.char_count == len(result.data.text)

    def test_gemini_schema_structure(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)
        schema = tool.to_gemini_schema()

        assert schema["name"] == "extract_section"
        assert "parameters" in schema
        props = schema["parameters"].get("properties", {})
        assert "paper_id" in props
        assert "section" in props

    def test_all_three_papers_extractable(self, section_store):
        tool = ExtractSectionTool(section_store=section_store)

        for paper_id in ["attention", "bert", "rag"]:
            result = tool.run({"paper_id": paper_id, "section": "abstract"})
            assert result.success is True, f"Failed for paper_id={paper_id}"
