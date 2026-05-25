from __future__ import annotations

import re
import unicodedata

from models import Chunk
from rag.embedding_core import EmbeddingModel
from vectorstore.fake_store import FakeVectorStore


class Retriever:
    _STOPWORDS = {
        "a",
        "as",
        "o",
        "os",
        "e",
        "de",
        "da",
        "das",
        "do",
        "dos",
        "em",
        "no",
        "nos",
        "na",
        "nas",
        "um",
        "uma",
        "quais",
        "qual",
        "que",
        "sao",
        "são",
        "sobre",
    }

    def __init__(self, embedding_model: EmbeddingModel, vector_store: FakeVectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int, filters: dict | None = None) -> list[Chunk]:
        query_embedding = self.embedding_model.embed(question)
        candidates = self.vector_store.search(
            query_embedding=query_embedding, top_k=top_k, filters=filters
        )
        fetch = getattr(self.vector_store, "fetch", None)
        if fetch:
            candidates = self._merge_candidates(candidates, fetch(filters=filters))
        scored: list[Chunk] = []
        for chunk in candidates:
            metadata = dict(chunk.metadata or {})
            metadata["hybrid_score"] = self._hybrid_score(question, chunk)
            scored.append(Chunk(chunk_id=chunk.chunk_id, text=chunk.text, metadata=metadata))
        reranked = sorted(scored, key=lambda chunk: chunk.metadata["hybrid_score"], reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _hybrid_score(question: str, chunk: Chunk) -> float:
        vec_score = float(chunk.metadata.get("retrieval_score", 0.0))
        q_tokens = Retriever._tokenize(question)
        c_tokens = Retriever._tokenize(chunk.text)
        if not q_tokens:
            lexical = 0.0
        else:
            lexical = len(q_tokens & c_tokens) / len(q_tokens)
        blended = (0.75 * vec_score) + (0.25 * lexical)
        return max(blended, lexical)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", text.lower())
        ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
        return {
            token
            for token in re.findall(r"\w+", ascii_text)
            if len(token) > 1 and token not in Retriever._STOPWORDS
        }

    @staticmethod
    def _merge_candidates(primary: list[Chunk], secondary: list[Chunk]) -> list[Chunk]:
        merged: dict[str, Chunk] = {}
        for chunk in [*primary, *secondary]:
            existing = merged.get(chunk.chunk_id)
            if not existing:
                merged[chunk.chunk_id] = chunk
                continue
            metadata = dict(chunk.metadata or {})
            metadata.update(existing.metadata or {})
            merged[chunk.chunk_id] = Chunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=metadata,
            )
        return list(merged.values())
