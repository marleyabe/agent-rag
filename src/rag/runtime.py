from __future__ import annotations

import os
import tempfile
from pathlib import Path

from citations.formatter import CitationFormatter
from config import AppConfig
from dotenv import load_dotenv
from ingestion.chunker import Chunker
from ingestion.loaders import DocumentLoader
from models import Answer, DocumentRecord
from rag.answer_generator import AnswerGenerator
from rag.embedding_core import GeminiEmbeddingModel
from rag.llm_core import GeminiLLM
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from storage.db import AppDatabase
from storage.files import FileStorage
from vectorstore.chroma_store import ChromaVectorStore


def _collection_name_from_env() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "chunks_fake_embedding_v1"
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    sanitized = model.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"chunks_{sanitized}"


class NotebookService:
    def __init__(self, config: AppConfig) -> None:
        load_dotenv()
        self.config = config
        self.db = AppDatabase(config.db_path)
        self.files = FileStorage(config.files_dir)
        self.embedding = GeminiEmbeddingModel()
        self.vector_store = ChromaVectorStore(
            str(config.chroma_dir),
            self.embedding,
            collection=_collection_name_from_env(),
        )
        self.pipeline = RagPipeline(
            document_loader=DocumentLoader(),
            chunker=Chunker(),
            vector_store=self.vector_store,
            retriever=Retriever(embedding_model=self.embedding, vector_store=self.vector_store),
            answer_generator=AnswerGenerator(
                llm=GeminiLLM(),
                citation_formatter=CitationFormatter(),
            ),
        )

    def ingest_uploaded_file(self, file_name: str, content: bytes) -> DocumentRecord:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        saved_path = self.files.save_upload(temp_path, file_name)
        record = self.pipeline.ingest(saved_path)
        self.db.upsert_document(record)
        temp_path.unlink(missing_ok=True)
        return record

    def ask(self, question: str, file_id: str | None = None) -> Answer:
        filters = {"file_id": file_id} if file_id else None
        answer = self.pipeline.ask(question, filters=filters)
        self.db.save_answer(question, answer)
        return answer
