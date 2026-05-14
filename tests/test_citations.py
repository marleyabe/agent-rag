from __future__ import annotations

from citations.formatter import CitationFormatter
from models import Chunk


def test_format_pdf_citation_with_predictable_link() -> None:
    chunk = Chunk(
        chunk_id="doc123_p2_c0",
        text="A multa por atraso e de 2%",
        metadata={
            "file_id": "doc123",
            "file_name": "exemplo.pdf",
            "file_type": "pdf",
            "page_number": 2,
            "paragraph_number": None,
            "start_line": 10,
            "end_line": 18,
            "chunk_id": "doc123_p2_c0",
        },
    )
    citation = CitationFormatter().format(chunk)
    assert citation.file_name == "exemplo.pdf"
    assert citation.page_number == 2
    assert citation.url == "/documents/doc123?page=2&chunk=doc123_p2_c0"


def test_format_docx_citation_with_predictable_link() -> None:
    chunk = Chunk(
        chunk_id="doc999_par3_c1",
        text="Pagamento em ate 10 dias",
        metadata={
            "file_id": "doc999",
            "file_name": "exemplo.docx",
            "file_type": "docx",
            "page_number": None,
            "paragraph_number": 3,
            "start_line": 1,
            "end_line": 3,
            "chunk_id": "doc999_par3_c1",
        },
    )
    citation = CitationFormatter().format(chunk)
    assert citation.paragraph_number == 3
    assert citation.url == "/documents/doc999?paragraph=3&chunk=doc999_par3_c1"
