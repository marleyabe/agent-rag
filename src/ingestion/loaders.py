from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
from docx import Document as DocxDocument

from models import DocumentSection


class DocumentLoader:
    def load(self, file_path: Path) -> list[DocumentSection]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        if suffix == ".docx":
            return self._load_docx(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _file_id(self, file_path: Path) -> str:
        digest = hashlib.sha1(str(file_path.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"doc_{digest}"

    def _load_pdf(self, file_path: Path) -> list[DocumentSection]:
        file_id = self._file_id(file_path)
        sections: list[DocumentSection] = []
        doc = fitz.open(file_path)
        try:
            for idx, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                sections.append(
                    DocumentSection(
                        file_id=file_id,
                        file_name=file_path.name,
                        file_type="pdf",
                        text=text,
                        page_number=idx,
                        paragraph_number=None,
                    )
                )
        finally:
            doc.close()
        return sections

    def _load_docx(self, file_path: Path) -> list[DocumentSection]:
        file_id = self._file_id(file_path)
        sections: list[DocumentSection] = []
        doc = DocxDocument(file_path)
        for idx, paragraph in enumerate(doc.paragraphs, start=1):
            text = paragraph.text.strip()
            if not text:
                continue
            sections.append(
                DocumentSection(
                    file_id=file_id,
                    file_name=file_path.name,
                    file_type="docx",
                    text=text,
                    page_number=None,
                    paragraph_number=idx,
                )
            )
        return sections
