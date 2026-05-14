from __future__ import annotations

from citations.formatter import CitationFormatter
from models import Chunk
from rag.answer_generator import AnswerGenerator
from rag.fake_llm import FakeLLM


def test_generate_answer_and_citations_without_hallucinated_sources() -> None:
    chunks = [
        Chunk(
            chunk_id="doc1_p1_c0",
            text="A multa por atraso e de 2%",
            metadata={
                "file_id": "doc1",
                "file_name": "contrato.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "paragraph_number": None,
                "start_line": 2,
                "end_line": 4,
                "chunk_id": "doc1_p1_c0",
            },
        )
    ]

    generator = AnswerGenerator(llm=FakeLLM(), citation_formatter=CitationFormatter())
    answer = generator.generate("qual a multa?", chunks)

    assert "2%" in answer.answer
    assert len(answer.citations) == 1
    assert answer.citations[0].file_name == "contrato.pdf"
    assert answer.citations[0].url.endswith("chunk=doc1_p1_c0")


def test_generate_answer_without_chunks_returns_no_information() -> None:
    generator = AnswerGenerator(llm=FakeLLM(), citation_formatter=CitationFormatter())
    answer = generator.generate("qual a multa?", [])
    assert "Nao encontrei" in answer.answer
    assert answer.citations == []
