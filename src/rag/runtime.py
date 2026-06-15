from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from citations.formatter import CitationFormatter
from config import AppConfig
from dotenv import load_dotenv
from ingestion.chunker import Chunker
from ingestion.loaders import DocumentLoader
from models import Answer, DocumentRecord
from rag.answer_generator import AnswerGenerator
from rag.embedding_core import OpenAIEmbeddingModel
from rag.llm_core import OpenAILLM
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from storage.db import AppDatabase
from storage.files import FileStorage
from vectorstore.chroma_store import ChromaVectorStore
from vectorstore.fake_store import FakeVectorStore


def _collection_name_from_env() -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "chunks_fake_embedding_v1"
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    sanitized = model.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"chunks_{sanitized}"


def _build_vector_store(config: AppConfig, embedding: OpenAIEmbeddingModel):
    try:
        return ChromaVectorStore(
            str(config.chroma_dir),
            embedding,
            collection=_collection_name_from_env(),
        )
    except Exception:
        return FakeVectorStore(embedding)


class NotebookService:
    def __init__(self, config: AppConfig) -> None:
        load_dotenv()
        self.config = config
        self.db = AppDatabase(config.db_path)
        self.files = FileStorage(config.files_dir)
        self.embedding = OpenAIEmbeddingModel()
        self.vector_store = _build_vector_store(config, self.embedding)
        self.pipeline = RagPipeline(
            document_loader=DocumentLoader(),
            chunker=Chunker(),
            vector_store=self.vector_store,
            retriever=Retriever(embedding_model=self.embedding, vector_store=self.vector_store),
            answer_generator=AnswerGenerator(
                llm=OpenAILLM(),
                citation_formatter=CitationFormatter(),
            ),
        )

    def ingest_uploaded_file(
        self,
        file_name: str,
        content: bytes,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> DocumentRecord:
        if progress_callback:
            progress_callback(0.05, "Preparando upload...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        if progress_callback:
            progress_callback(0.2, "Salvando arquivo...")
        saved_path = self.files.save_upload(temp_path, file_name)
        if progress_callback:
            progress_callback(0.45, "Extraindo e segmentando texto...")
        record = self.pipeline.ingest(saved_path)
        if progress_callback:
            progress_callback(0.9, "Persistindo metadados...")
        self.db.upsert_document(record)
        temp_path.unlink(missing_ok=True)
        if progress_callback:
            progress_callback(1.0, "Concluido.")
        return record

    def ask(self, question: str, file_id: str | None = None) -> Answer:
        filters = {"file_id": file_id} if file_id else None
        answer = self.pipeline.ask(question, filters=filters)
        self.db.save_answer(question, answer)
        return answer
