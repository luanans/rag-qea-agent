import json

from tools.list_sections import ListSectionsTool


class TestListSectionsTool:
    def test_list_sections_for_one_paper(self, section_store):
        tool = ListSectionsTool(section_store=section_store)
        result = tool.run({"paper_id": "attention"})

        assert result.success is True
        data = json.loads(result.output)
        assert len(data["papers"]) == 1
        assert data["papers"][0]["paper_id"] == "attention"
        assert "abstract" in data["papers"][0]["sections"]

    def test_list_sections_for_all_papers(self, section_store):
        tool = ListSectionsTool(section_store=section_store)
        result = tool.run({})

        assert result.success is True
        data = json.loads(result.output)
        paper_ids = [p["paper_id"] for p in data["papers"]]
        assert "attention" in paper_ids
        assert "bert" in paper_ids
        assert "rag" in paper_ids

    def test_sections_include_introduction(self, section_store):
        tool = ListSectionsTool(section_store=section_store)
        result = tool.run({"paper_id": "attention"})

        data = json.loads(result.output)
        sections = data["papers"][0]["sections"]
        assert "introduction" in sections

    def test_invalid_paper_id_returns_failure(self, section_store):
        tool = ListSectionsTool(section_store=section_store)
        result = tool.run({"paper_id": "gpt4"})

        assert result.success is False

    def test_gemini_schema_structure(self, section_store):
        tool = ListSectionsTool(section_store=section_store)
        schema = tool.to_gemini_schema()

        assert schema["name"] == "list_sections"
        assert "parameters" in schema
        props = schema["parameters"].get("properties", {})
        assert "paper_id" in props
