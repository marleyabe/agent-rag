from __future__ import annotations

from models import DocumentRecord
from rag.runtime import NotebookService


def ingest_upload(service: NotebookService, file_name: str, content: bytes) -> DocumentRecord:
    return service.ingest_uploaded_file(file_name, content)

