from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from ingestion.chunker import Chunker
from ingestion.loaders import DocumentLoader
from models import Answer, Chunk, DocumentRecord
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
        self._default_top_k = 20
        self._final_context_k = 6
        self._min_retrieval_score = float(os.getenv("RAG_MIN_RETRIEVAL_SCORE", "0.15"))
        self._overview_context_k = 8

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
        if self._is_overview_question(question):
            overview_chunks = self._overview_chunks(filters=filters)
            if overview_chunks:
                return self.answer_generator.generate(question=question, chunks=overview_chunks)

        retrieved = self.retriever.retrieve(question=question, top_k=self._default_top_k, filters=filters)
        if not retrieved:
            return self.answer_generator.generate(question=question, chunks=[])

        scored = [self._confidence_score(chunk) for chunk in retrieved]
        max_score = max(scored)
        if max_score < self._min_retrieval_score:
            return self.answer_generator.generate(question=question, chunks=[])

        filtered = [
            chunk
            for chunk in retrieved
            if self._confidence_score(chunk) >= self._min_retrieval_score
        ]
        context_chunks = (filtered or retrieved)[: self._final_context_k]
        return self.answer_generator.generate(question=question, chunks=context_chunks)

    def _overview_chunks(self, filters: dict | None) -> list[Chunk]:
        fetch = getattr(self.vector_store, "fetch", None)
        if not fetch:
            return []
        chunks = fetch(filters=filters)
        ordered = sorted(chunks, key=self._source_order)
        if not ordered:
            return []

        front_matter = ordered[: self._overview_context_k]
        summary_chunks = [
            chunk
            for chunk in ordered
            if self._contains_any(chunk.text, ("sumario", "indice", "titulo i", "preambulo"))
        ][: self._overview_context_k]

        selected: list[Chunk] = []
        seen: set[str] = set()
        for chunk in [*front_matter, *summary_chunks]:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            selected.append(chunk)
            if len(selected) >= self._overview_context_k:
                break
        return selected

    @staticmethod
    def _confidence_score(chunk: Chunk) -> float:
        return float(chunk.metadata.get("hybrid_score", chunk.metadata.get("retrieval_score", 0.0)))

    @staticmethod
    def _source_order(chunk: Chunk) -> tuple[int, int, int, int]:
        metadata = chunk.metadata or {}
        page = metadata.get("page_number") or 0
        paragraph = metadata.get("paragraph_number") or 0
        start_line = metadata.get("start_line") or 0
        chunk_index = metadata.get("chunk_index") or 0
        return int(page), int(paragraph), int(start_line), int(chunk_index)

    @staticmethod
    def _is_overview_question(question: str) -> bool:
        normalized = RagPipeline._normalize(question)
        overview_patterns = (
            r"\bo que (o|este|esse)?\s*documento (fala|diz|trata|aborda)",
            r"\bdo que (o|este|esse)?\s*documento (fala|trata)",
            r"\bresum[ao]\b",
            r"\bvisao geral\b",
            r"\bassunto (principal|central)\b",
        )
        return any(re.search(pattern, normalized) for pattern in overview_patterns)

    @staticmethod
    def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
        normalized = RagPipeline._normalize(text)
        return any(needle in normalized for needle in needles)

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return "".join(char for char in normalized if not unicodedata.combining(char))
