from __future__ import annotations

import sys
import types

import pytest

from models import Chunk
from rag.embedding_core import FakeEmbeddingModel, GeminiEmbeddingModel
from rag.fake_llm import LLM
from rag.llm_core import GeminiLLM
from vectorstore.chroma_store import ChromaVectorStore


def test_llm_base_interface_raises() -> None:
    with pytest.raises(NotImplementedError):
        LLM().generate("q", [])


def test_gemini_embedding_fallback_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    emb = GeminiEmbeddingModel()
    assert emb.embed("multa contrato") == [1.0, 1.0, 0.0, 0.0]


def test_gemini_embedding_with_mocked_client(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "m")

    class FakeResp:
        embeddings = [types.SimpleNamespace(values=[0.2, 0.3])]

    class FakeModels:
        def embed_content(self, model: str, contents: str) -> FakeResp:
            assert model == "m"
            assert "abc" in contents
            return FakeResp()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "x"
            self.models = FakeModels()

    fake_google = types.SimpleNamespace(genai=types.SimpleNamespace(Client=FakeClient))
    monkeypatch.setitem(sys.modules, "google", fake_google)
    emb = GeminiEmbeddingModel()
    assert emb.embed("abc") == [0.2, 0.3]


def test_llama_llm_fallback_and_mocked_client(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm = GeminiLLM()
    assert "Nao encontrei" in llm.generate("q", [])

    chunk = Chunk("c1", "texto base", {"file_id": "f", "chunk_id": "c1"})
    assert "documentos enviados" in llm.generate("q", [chunk])

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-flash-preview")

    class FakeModels:
        def generate_content(self, model: str, contents: str):
            assert model == "gemini-3-flash-preview"
            assert "Pergunta:" in contents
            return types.SimpleNamespace(text="resposta gemini")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "k"
            self.models = FakeModels()

    fake_google = types.SimpleNamespace(genai=types.SimpleNamespace(Client=FakeClient))
    monkeypatch.setitem(sys.modules, "google", fake_google)
    llm2 = GeminiLLM()
    assert llm2.generate("q", [chunk]) == "resposta gemini"


def test_chroma_store_add_and_search_with_mocked_client(monkeypatch, tmp_path) -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.saved = None

        def upsert(self, **kwargs):
            self.saved = kwargs

        def query(self, **kwargs):
            assert kwargs["where"] == {"file_id": "f1"}
            return {
                "ids": [["c1"]],
                "documents": [["multa"]],
                "metadatas": [[{"file_id": "f1", "chunk_id": "c1"}]],
            }

    class FakeClient:
        def __init__(self, path: str) -> None:
            self.path = path
            self.col = FakeCollection()

        def get_or_create_collection(self, name: str):
            assert name == "chunks"
            return self.col

    monkeypatch.setattr("vectorstore.chroma_store.PersistentClient", FakeClient)
    emb = FakeEmbeddingModel()
    store = ChromaVectorStore(str(tmp_path), emb)
    store.add([])
    chunks = [Chunk("c1", "multa", {"file_id": "f1", "chunk_id": "c1"})]
    store.add(chunks)
    assert store.collection.saved["ids"] == ["c1"]

    out = store.search([1.0, 0, 0, 0], top_k=1, filters={"file_id": "f1"})
    assert len(out) == 1 and out[0].chunk_id == "c1"
