from __future__ import annotations

from typing import Callable

from models import DocumentRecord
from rag.runtime import NotebookService


def ingest_upload(
    service: NotebookService,
    file_name: str,
    content: bytes,
    progress_callback: Callable[[float, str], None] | None = None,
) -> DocumentRecord:
    return service.ingest_uploaded_file(file_name, content, progress_callback=progress_callback)
