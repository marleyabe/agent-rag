from __future__ import annotations

from rag.embedding_core import FakeEmbeddingModel
from rag.llm_core import OpenAILLM
from models import Chunk


def test_embedding_core_fake_is_deterministic() -> None:
    emb = FakeEmbeddingModel()
    assert emb.embed("multa contrato") == [1.0, 1.0, 0.0, 0.0]


def test_llm_core_fallback_without_chunks() -> None:
    llm = OpenAILLM()
    assert "Nao encontrei" in llm.generate("q", [])
    out = llm.generate("q", [Chunk("c1", "texto", {"file_id": "f1", "chunk_id": "c1"})])
    assert isinstance(out, str) and len(out) > 0
