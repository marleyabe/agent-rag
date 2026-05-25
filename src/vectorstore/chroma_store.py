from __future__ import annotations

from chromadb import PersistentClient

from models import Chunk
from rag.embedding_core import EmbeddingModel


class ChromaVectorStore:
    def __init__(self, persist_dir: str, embedding_model: EmbeddingModel, collection: str = "chunks") -> None:
        self.embedding_model = embedding_model
        self.client = PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        docs = [chunk.text for chunk in chunks]
        metas = [chunk.metadata for chunk in chunks]
        embeds = self.embedding_model.embed_many(docs)
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    def search(
        self, query_embedding: list[float], top_k: int, filters: dict | None = None
    ) -> list[Chunk]:
        where = filters or None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        ids = result.get("ids", [[]])[-1]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[Chunk] = []
        for idx, (chunk_id, text, meta) in enumerate(zip(ids, docs, metas)):
            metadata = dict(meta or {})
            if idx < len(distances):
                distance = float(distances[idx])
                metadata["retrieval_distance"] = distance
                metadata["retrieval_score"] = 1.0 / (1.0 + distance)
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))
        return chunks

    def fetch(self, filters: dict | None = None, limit: int | None = None) -> list[Chunk]:
        result = self.collection.get(
            where=filters or None,
            limit=limit,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        return [
            Chunk(chunk_id=chunk_id, text=text, metadata=dict(meta or {}))
            for chunk_id, text, meta in zip(ids, docs, metas)
        ]
