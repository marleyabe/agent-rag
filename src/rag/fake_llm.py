from __future__ import annotations

from models import Chunk


class LLM:
    def generate(self, question: str, chunks: list[Chunk]) -> str:
        raise NotImplementedError


class FakeLLM(LLM):
    def generate(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "Nao encontrei essa informacao nos documentos enviados."
        return f"{chunks[0].text}, conforme os documentos enviados."
