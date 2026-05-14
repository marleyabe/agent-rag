from __future__ import annotations

from citations.links import build_document_url
from models import Chunk, Citation


class CitationFormatter:
    def format(self, chunk: Chunk) -> Citation:
        md = chunk.metadata
        file_id = md["file_id"]
        chunk_id = md.get("chunk_id", chunk.chunk_id)
        page_number = md.get("page_number")
        paragraph_number = md.get("paragraph_number")
        url = build_document_url(
            file_id=file_id,
            chunk_id=chunk_id,
            page_number=page_number,
            paragraph_number=paragraph_number,
        )
        return Citation(
            file_name=md.get("file_name", ""),
            page_number=page_number,
            paragraph_number=paragraph_number,
            start_line=md.get("start_line"),
            end_line=md.get("end_line"),
            source_text=chunk.text,
            url=url,
        )
