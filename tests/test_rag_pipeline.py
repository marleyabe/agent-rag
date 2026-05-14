from __future__ import annotations

from pathlib import Path

from citations.formatter import CitationFormatter
from ingestion.chunker import Chunker
from ingestion.loaders import DocumentLoader
from rag.answer_generator import AnswerGenerator
from rag.embedding_core import FakeEmbeddingModel
from rag.fake_llm import FakeLLM
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from vectorstore.fake_store import FakeVectorStore


def test_rag_pipeline_end_to_end_with_citations(sample_pdf: Path) -> None:
    embedding = FakeEmbeddingModel()
    store = FakeVectorStore(embedding_model=embedding)
    pipeline = RagPipeline(
        document_loader=DocumentLoader(),
        chunker=Chunker(max_lines=6),
        vector_store=store,
        retriever=Retriever(embedding_model=embedding, vector_store=store),
        answer_generator=AnswerGenerator(llm=FakeLLM(), citation_formatter=CitationFormatter()),
    )

    record = pipeline.ingest(sample_pdf)
    answer = pipeline.ask("Qual e a multa por atraso?", filters={"file_id": record.file_id})

    assert record.file_name == "sample.pdf"
    assert answer.citations
    assert "documentos enviados" in answer.answer.lower()
