from __future__ import annotations

from pathlib import Path

from models import Chunk
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


def test_pipeline_returns_safe_fallback_when_retrieval_is_low_confidence(monkeypatch) -> None:
    class LowScoreRetriever:
        def retrieve(self, question: str, top_k: int, filters: dict | None = None) -> list[Chunk]:
            return [
                Chunk(
                    chunk_id="c1",
                    text="texto irrelevante",
                    metadata={"retrieval_score": 0.01, "file_id": "f1", "chunk_id": "c1"},
                )
            ]

    embedding = FakeEmbeddingModel()
    store = FakeVectorStore(embedding_model=embedding)
    monkeypatch.setenv("RAG_MIN_RETRIEVAL_SCORE", "0.15")
    pipeline = RagPipeline(
        document_loader=DocumentLoader(),
        chunker=Chunker(max_lines=6),
        vector_store=store,
        retriever=LowScoreRetriever(),
        answer_generator=AnswerGenerator(llm=FakeLLM(), citation_formatter=CitationFormatter()),
    )
    answer = pipeline.ask("pergunta sem base")
    assert "Nao encontrei essa informacao" in answer.answer


def test_pipeline_answers_overview_question_from_document_front_matter(sample_pdf: Path) -> None:
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
    answer = pipeline.ask("O que o documento fala?", filters={"file_id": record.file_id})

    assert "Contrato de servico" in answer.answer
    assert answer.citations


def test_pipeline_uses_hybrid_score_for_low_confidence_gate(sample_pdf: Path) -> None:
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
    answer = pipeline.ask("inadimplencia", filters={"file_id": record.file_id})

    assert "Nao encontrei essa informacao" not in answer.answer
