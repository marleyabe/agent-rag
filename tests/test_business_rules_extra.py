from __future__ import annotations

from citations.formatter import CitationFormatter
from ingestion.chunker import Chunker
from models import Chunk, DocumentSection
from rag.answer_generator import AnswerGenerator
from rag.embedding_core import FakeEmbeddingModel
from rag.fake_llm import FakeLLM
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from vectorstore.fake_store import FakeVectorStore


def test_chunker_handles_empty_section_and_docx_id() -> None:
    sections = [
        DocumentSection("f", "a.docx", "docx", "", None, 2),
        DocumentSection("f", "a.docx", "docx", "linha1\nlinha2", None, 2),
    ]
    chunks = Chunker(max_lines=10).split(sections)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "f_par2_c0"


def test_answer_generator_deduplicates_and_caps_five_citations() -> None:
    chunk = Chunk(
        "c1",
        "texto",
        {
            "file_id": "f1",
            "file_name": "a.pdf",
            "file_type": "pdf",
            "page_number": 1,
            "paragraph_number": None,
            "start_line": 1,
            "end_line": 2,
            "chunk_id": "c1",
        },
    )
    # 7 entradas com duplicatas da mesma url/source_text
    chunks = [chunk, chunk, chunk, chunk, chunk, chunk, chunk]
    ans = AnswerGenerator(FakeLLM(), CitationFormatter()).generate("q", chunks)
    assert len(ans.citations) == 1


def test_pipeline_ingest_with_no_sections_returns_file_stem() -> None:
    class EmptyLoader:
        def load(self, _):  # noqa: ANN001
            return []

    emb = FakeEmbeddingModel()
    store = FakeVectorStore(emb)
    pipeline = RagPipeline(
        document_loader=EmptyLoader(),  # type: ignore[arg-type]
        chunker=Chunker(),
        vector_store=store,
        retriever=Retriever(emb, store),
        answer_generator=AnswerGenerator(FakeLLM(), CitationFormatter()),
    )
    rec = pipeline.ingest(__import__("pathlib").Path("/tmp/x.pdf"))
    assert rec.file_id == "x"
