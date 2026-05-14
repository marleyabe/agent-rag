from __future__ import annotations

from citations.formatter import CitationFormatter
from models import Answer, Chunk
from rag.fake_llm import LLM


class AnswerGenerator:
    def __init__(self, llm: LLM, citation_formatter: CitationFormatter) -> None:
        self.llm = llm
        self.citation_formatter = citation_formatter

    def generate(self, question: str, chunks: list[Chunk]) -> Answer:
        if not chunks:
            return Answer(
                answer="Nao encontrei essa informacao nos documentos enviados.",
                citations=[],
            )
        answer_text = self.llm.generate(question, chunks)
        citations = [self.citation_formatter.format(chunk) for chunk in chunks]
        unique: list = []
        seen = set()
        for citation in citations:
            key = (citation.url, citation.source_text)
            if key in seen:
                continue
            seen.add(key)
            unique.append(citation)
            if len(unique) >= 5:
                break
        return Answer(answer=answer_text, citations=unique)
