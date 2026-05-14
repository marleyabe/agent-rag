from __future__ import annotations

from pathlib import Path
from typing import Any

from storage.db import AppDatabase
from ui.viewer import build_citation_lookup_url


def get_viewer_payload(
    db: AppDatabase,
    file_id: str,
    chunk: str,
    page: str = "",
    paragraph: str = "",
) -> dict[str, Any]:
    citation_url = build_citation_lookup_url(
        file_id=file_id,
        chunk=chunk,
        page=page,
        paragraph=paragraph,
    )
    citation = db.get_citation_by_url(citation_url)
    document = db.get_document(file_id)
    return {"citation_url": citation_url, "citation": citation, "document": document}


def pdf_anchor_download_name(file_name: str, page: int | None) -> str:
    if page is None:
        return file_name
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix or ".pdf"
    return f"{stem}_page_{page}{suffix}"


def build_pdf_open_url(file_path: str, page: int | None) -> str:
    if page is None:
        return file_path
    return f"{file_path}#page={page}"
