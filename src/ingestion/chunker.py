from __future__ import annotations

from models import Chunk, DocumentSection


class Chunker:
    def __init__(self, max_chars: int = 800, max_lines: int = 20) -> None:
        self.max_chars = max_chars
        self.max_lines = max_lines

    def split(self, sections: list[DocumentSection]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            lines = [line.strip() for line in section.text.splitlines() if line.strip()]
            if not lines:
                continue

            current_lines: list[str] = []
            chunk_index = 0
            start_line = 1
            current_chars = 0

            def flush(end_line: int) -> None:
                nonlocal chunk_index, current_lines, current_chars, start_line
                if not current_lines:
                    return
                text = "\n".join(current_lines)
                chunk_id = self._build_chunk_id(section, chunk_index)
                metadata = {
                    "file_id": section.file_id,
                    "file_name": section.file_name,
                    "file_type": section.file_type,
                    "page_number": section.page_number,
                    "paragraph_number": section.paragraph_number,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "start_line": start_line,
                    "end_line": end_line,
                    "source_text": text,
                }
                chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))
                chunk_index += 1
                current_lines = []
                current_chars = 0
                start_line = end_line + 1

            for idx, line in enumerate(lines, start=1):
                projected = current_chars + len(line) + (1 if current_lines else 0)
                if current_lines and (len(current_lines) >= self.max_lines or projected > self.max_chars):
                    flush(idx - 1)
                current_lines.append(line)
                current_chars += len(line) + (1 if len(current_lines) > 1 else 0)
            flush(len(lines))

        return chunks

    @staticmethod
    def _build_chunk_id(section: DocumentSection, chunk_index: int) -> str:
        if section.file_type == "pdf":
            return f"{section.file_id}_p{section.page_number}_c{chunk_index}"
        return f"{section.file_id}_par{section.paragraph_number}_c{chunk_index}"
