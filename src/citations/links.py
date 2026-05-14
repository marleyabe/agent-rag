from __future__ import annotations


def build_document_url(
    file_id: str,
    chunk_id: str,
    page_number: int | None = None,
    paragraph_number: int | None = None,
) -> str:
    if page_number is not None:
        return f"/documents/{file_id}?page={page_number}&chunk={chunk_id}"
    return f"/documents/{file_id}?paragraph={paragraph_number}&chunk={chunk_id}"

