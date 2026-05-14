from __future__ import annotations

import os


class EmbeddingModel:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class FakeEmbeddingModel(EmbeddingModel):
    def embed(self, text: str) -> list[float]:
        normalized = text.lower()
        keywords = ["multa", "contrato", "prazo", "pagamento"]
        return [1.0 if token in normalized else 0.0 for token in keywords]


class GeminiEmbeddingModel(EmbeddingModel):
    def __init__(self) -> None:
        self._fallback = FakeEmbeddingModel()
        self._client = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        self._model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    def embed(self, text: str) -> list[float]:
        if self._client is None:
            return self._fallback.embed(text)
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(response.embeddings[0].values)

