from __future__ import annotations

import os

from models import Chunk
from rag.fake_llm import LLM, FakeLLM


class GeminiLLM(LLM):
    def __init__(self) -> None:
        self._fallback = FakeLLM()
        self._client = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        self._model = os.getenv("GEMINI_MODEL", "models/gemini-3-flash-preview")

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "Nao encontrei essa informacao nos documentos enviados."
        if self._client is None:
            return self._fallback.generate(question, chunks)
        context = "\n\n".join([chunk.text for chunk in chunks[:5]])
        prompt = (
            "Responda em portugues com base apenas no contexto abaixo. "
            "Se faltar informacao, diga que nao encontrou nos documentos.\n\n"
            f"Pergunta: {question}\n\nContexto:\n{context}"
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return (response.text or "").strip()

