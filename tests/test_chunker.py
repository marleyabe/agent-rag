from __future__ import annotations

from ingestion.chunker import Chunker
from models import DocumentSection


def _section() -> DocumentSection:
    return DocumentSection(
        file_id="doc-1",
        file_name="sample.pdf",
        file_type="pdf",
        text=(
            "linha1\nlinha2 multa\nlinha3 contrato\nlinha4 prazo\n"
            "linha5 pagamento\nlinha6\nlinha7\nlinha8\nlinha9\nlinha10"
        ),
        page_number=1,
        paragraph_number=None,
    )


def test_split_text_into_stable_chunks_and_preserve_metadata() -> None:
    chunker = Chunker(max_lines=4, overlap_lines=0)
    chunks_a = chunker.split([_section()])
    chunks_b = chunker.split([_section()])

    assert [chunk.text for chunk in chunks_a] == [chunk.text for chunk in chunks_b]
    assert len(chunks_a) == 3
    assert chunks_a[0].metadata["file_id"] == "doc-1"
    assert chunks_a[0].metadata["file_name"] == "sample.pdf"
    assert chunks_a[0].metadata["page_number"] == 1
    assert chunks_a[0].metadata["chunk_index"] == 0
    assert chunks_a[0].metadata["start_line"] == 1
    assert chunks_a[0].metadata["end_line"] == 4


def test_respects_approximate_max_chunk_size() -> None:
    chunker = Chunker(max_chars=50, max_lines=100, overlap_lines=0)
    chunks = chunker.split([_section()])
    assert all(len(chunk.text) <= 70 for chunk in chunks)


def test_generates_overlap_between_chunks() -> None:
    chunker = Chunker(max_lines=4, overlap_lines=2)
    chunks = chunker.split([_section()])
    assert len(chunks) >= 2
    first_tail = chunks[0].text.splitlines()[-2:]
    second_head = chunks[1].text.splitlines()[:2]
    assert first_tail == second_head
