from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentRecord:
    file_id: str
    file_name: str
    file_type: str
    file_path: str


@dataclass
class DocumentSection:
    file_id: str
    file_name: str
    file_type: str
    text: str
    page_number: int | None
    paragraph_number: int | None


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


@dataclass
class Citation:
    file_name: str
    page_number: int | None
    paragraph_number: int | None
    start_line: int | None
    end_line: int | None
    source_text: str
    url: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation]
