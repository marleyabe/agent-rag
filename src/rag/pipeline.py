from __future__ import annotations

from pathlib import Path

from ingestion.chunker import Chunker
from ingestion.loaders import DocumentLoader
from models import Answer, DocumentRecord
from rag.answer_generator import AnswerGenerator
from rag.retriever import Retriever
from vectorstore.fake_store import FakeVectorStore


class RagPipeline:
    def __init__(
        self,
        document_loader: DocumentLoader,
        chunker: Chunker,
        vector_store: FakeVectorStore,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
    ) -> None:
        self.document_loader = document_loader
        self.chunker = chunker
        self.vector_store = vector_store
        self.retriever = retriever
        self.answer_generator = answer_generator
        self._default_top_k = 5

    def ingest(self, file_path: Path) -> DocumentRecord:
        sections = self.document_loader.load(file_path)
        chunks = self.chunker.split(sections)
        self.vector_store.add(chunks)

        file_type = file_path.suffix.lstrip(".").lower()
        if sections:
            file_id = sections[0].file_id
            file_name = sections[0].file_name
            file_type = sections[0].file_type
        else:
            file_id = file_path.stem
            file_name = file_path.name

        return DocumentRecord(
            file_id=file_id,
            file_name=file_name,
            file_type=file_type,
            file_path=str(file_path),
        )

    def ask(self, question: str, filters: dict | None = None) -> Answer:
        chunks = self.retriever.retrieve(question=question, top_k=self._default_top_k, filters=filters)
        return self.answer_generator.generate(question=question, chunks=chunks)
