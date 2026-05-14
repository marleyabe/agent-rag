from __future__ import annotations

from models import Answer, DocumentRecord
from rag.runtime import NotebookService


def resolve_filter_file_id(only_last: bool, last_record: DocumentRecord | None) -> str | None:
    if only_last and last_record:
        return last_record.file_id
    return None


def ask_question(service: NotebookService, question: str, file_id: str | None) -> Answer:
    return service.ask(question, file_id=file_id)

