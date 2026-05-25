from __future__ import annotations

import hashlib
import os


class EmbeddingModel:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class FakeEmbeddingModel(EmbeddingModel):
    def embed(self, text: str) -> list[float]:
        normalized = text.lower()
        keywords = ["multa", "contrato", "prazo", "pagamento"]
        return [1.0 if token in normalized else 0.0 for token in keywords]


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self) -> None:
        self._fallback = FakeEmbeddingModel()
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
        self._model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._batch_size = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "64"))
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._client is None:
            return self._fallback.embed_many(texts)

        results: list[list[float] | None] = [None] * len(texts)
        missing_map: dict[str, list[int]] = {}

        for idx, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self._cache.get(key)
            if cached is not None:
                results[idx] = cached
                continue
            missing_map.setdefault(text, []).append(idx)

        missing_texts = list(missing_map.keys())

        for start in range(0, len(missing_texts), self._batch_size):
            batch_texts = missing_texts[start : start + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch_texts,
            )
            for local_idx, emb_data in enumerate(response.data):
                embedding = list(emb_data.embedding)
                original_text = batch_texts[local_idx]
                self._cache[self._cache_key(original_text)] = embedding
                for original_idx in missing_map[original_text]:
                    results[original_idx] = embedding

        return [emb or self._fallback.embed(texts[idx]) for idx, emb in enumerate(results)]

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._model}:{digest}"
