from __future__ import annotations

import sys
import types

import pytest

from models import Chunk
from rag.embedding_core import FakeEmbeddingModel, OpenAIEmbeddingModel
from rag.fake_llm import LLM
from rag.llm_core import OpenAILLM
from vectorstore.chroma_store import ChromaVectorStore


def test_llm_base_interface_raises() -> None:
    with pytest.raises(NotImplementedError):
        LLM().generate("q", [])


def test_openai_embedding_fallback_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    emb = OpenAIEmbeddingModel()
    assert emb.embed("multa contrato") == [1.0, 1.0, 0.0, 0.0]


def test_openai_embedding_with_mocked_client(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "m")

    class FakeEmbeddings:
        def create(self, model: str, input: str):
            assert model == "m"
            assert "abc" in input
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.2, 0.3])]
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "x"
            self.embeddings = FakeEmbeddings()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    emb = OpenAIEmbeddingModel()
    assert emb.embed("abc") == [0.2, 0.3]


def test_openai_embedding_batch_and_cache(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "m")
    monkeypatch.setenv("OPENAI_EMBEDDING_BATCH_SIZE", "2")
    calls: list[list[str]] = []

    class FakeEmbeddings:
        def create(self, model: str, input: list[str]):
            assert model == "m"
            calls.append(list(input))
            data = [types.SimpleNamespace(embedding=[float(len(text)), 1.0]) for text in input]
            return types.SimpleNamespace(data=data)

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "x"
            self.embeddings = FakeEmbeddings()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    emb = OpenAIEmbeddingModel()
    out1 = emb.embed_many(["aa", "bbb", "aa"])
    out2 = emb.embed_many(["bbb", "cccc"])
    assert out1 == [[2.0, 1.0], [3.0, 1.0], [2.0, 1.0]]
    assert out2 == [[3.0, 1.0], [4.0, 1.0]]
    assert calls == [["aa", "bbb"], ["cccc"]]


def test_llama_llm_fallback_and_mocked_client(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm = OpenAILLM()
    assert "Nao encontrei" in llm.generate("q", [])

    chunk = Chunk("c1", "texto base", {"file_id": "f", "chunk_id": "c1"})
    assert "documentos enviados" in llm.generate("q", [chunk])

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, model: str, messages: list[dict[str, str]]):
            self.calls += 1
            assert model == "gpt-4.1-mini"
            assert "Pergunta:" in messages[1]["content"]
            if self.calls == 1:
                content = "Fato 1 [C1]"
            else:
                content = "Resposta final [C1]"
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
            )

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "k"
            self.chat = FakeChat()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    llm2 = OpenAILLM()
    assert llm2.generate("q", [chunk]) == "Resposta final [C1]"


def test_openai_llm_uses_extractive_fallback_when_model_reports_no_evidence(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")

    class FakeCompletions:
        def create(self, model: str, messages: list[dict[str, str]]):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="SEM_EVIDENCIA"))]
            )

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.chat = FakeChat()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    llm = OpenAILLM()
    chunk = Chunk("c1", "Dos Direitos e Garantias Fundamentais", {"file_id": "f", "chunk_id": "c1"})

    assert "Dos Direitos" in llm.generate("Quais sao os direitos fundamentais?", [chunk])


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
                "distances": [[0.25]],
            }

        def get(self, **kwargs):
            assert kwargs["where"] == {"file_id": "f1"}
            assert kwargs["limit"] == 1
            assert kwargs["include"] == ["documents", "metadatas"]
            return {
                "ids": ["c1"],
                "documents": ["multa"],
                "metadatas": [{"file_id": "f1", "chunk_id": "c1"}],
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
    assert out[0].metadata["retrieval_score"] == 0.8

    fetched = store.fetch(filters={"file_id": "f1"}, limit=1)
    assert fetched == [Chunk("c1", "multa", {"file_id": "f1", "chunk_id": "c1"})]
