from __future__ import annotations

import os

from models import Chunk
from rag.fake_llm import LLM, FakeLLM


class OpenAILLM(LLM):
    def __init__(self) -> None:
        self._fallback = FakeLLM()
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
        self._model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "Nao encontrei essa informacao nos documentos enviados."
        if self._client is None:
            return self._fallback.generate(question, chunks)
        context = "\n\n".join(
            [f"[C{idx + 1}] {chunk.text}" for idx, chunk in enumerate(chunks[:8])]
        )
        answer_prompt = (
            "Responda em portugues de forma objetiva usando SOMENTE o contexto abaixo. "
            "Toda afirmacao factual deve conter uma citacao [C#]. "
            "Se o contexto nao tiver fatos suficientes para responder, retorne exatamente: "
            "SEM_EVIDENCIA.\n\n"
            f"Pergunta: {question}\n\nContexto:\n{context}"
        )
        try:
            answer = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Voce gera respostas RAG estritamente fundamentadas em evidencias."
                        ),
                    },
                    {"role": "user", "content": answer_prompt},
                ],
            )
        except Exception:
            return self._fallback.generate(question, chunks)
        answer_text = (answer.choices[0].message.content or "").strip()
        if not answer_text or "SEM_EVIDENCIA" in answer_text:
            return self._fallback.generate(question, chunks)
        return answer_text
