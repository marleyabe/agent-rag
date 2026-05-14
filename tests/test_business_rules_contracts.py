from __future__ import annotations

from citations.formatter import CitationFormatter
from ingestion.chunker import Chunker
from models import Chunk, DocumentSection
from rag.answer_generator import AnswerGenerator
from rag.fake_llm import FakeLLM


def test_chunk_metadata_contract_contains_required_fields() -> None:
    sections = [
        DocumentSection(
            file_id="d1",
            file_name="doc.pdf",
            file_type="pdf",
            text="l1\nl2\nl3",
            page_number=1,
            paragraph_number=None,
        )
    ]
    chunks = Chunker(max_lines=2).split(sections)
    md = chunks[0].metadata
    required = {
        "file_id",
        "file_name",
        "file_type",
        "page_number",
        "paragraph_number",
        "chunk_id",
        "chunk_index",
        "start_line",
        "end_line",
        "source_text",
    }
    assert required.issubset(md.keys())


def test_answer_generator_caps_at_five_distinct_citations() -> None:
    chunks = []
    for i in range(7):
        chunks.append(
            Chunk(
                chunk_id=f"c{i}",
                text=f"texto {i}",
                metadata={
                    "file_id": "f1",
                    "file_name": "doc.pdf",
                    "file_type": "pdf",
                    "page_number": 1,
                    "paragraph_number": None,
                    "start_line": i + 1,
                    "end_line": i + 1,
                    "chunk_id": f"c{i}",
                },
            )
        )

    answer = AnswerGenerator(FakeLLM(), CitationFormatter()).generate("q", chunks)
    assert len(answer.citations) == 5
    urls = [citation.url for citation in answer.citations]
    assert len(urls) == len(set(urls))


def test_answer_without_chunks_never_returns_citations() -> None:
    answer = AnswerGenerator(FakeLLM(), CitationFormatter()).generate("q", [])
    assert answer.citations == []
    assert "Nao encontrei" in answer.answer

