from __future__ import annotations

from pathlib import Path

from config import AppConfig
from models import Answer, Citation, DocumentRecord
from rag.runtime import NotebookService, _collection_name_from_env
from storage.db import AppDatabase
from storage.files import FileStorage


def test_file_storage_save_upload(tmp_path: Path) -> None:
    src = tmp_path / "in.pdf"
    src.write_bytes(b"abc")
    storage = FileStorage(tmp_path / "files")
    dst = storage.save_upload(src, "saved.pdf")
    assert dst.exists()
    assert dst.read_bytes() == b"abc"


def test_app_database_document_and_answer_persistence(tmp_path: Path) -> None:
    db = AppDatabase(tmp_path / "app.db")
    record = DocumentRecord("f1", "a.pdf", "pdf", "/x/a.pdf")
    db.upsert_document(record)
    db.upsert_document(DocumentRecord("f1", "a2.pdf", "pdf", "/x/a2.pdf"))
    got = db.get_document("f1")
    assert got and got["file_name"] == "a2.pdf"
    assert db.get_document("missing") is None

    answer = Answer(
        answer="ok",
        citations=[
            Citation("a2.pdf", 1, None, 1, 2, "trecho", "/documents/f1?page=1&chunk=c1"),
        ],
    )
    chat_id = db.save_answer("q", answer)
    assert chat_id >= 1
    citation = db.get_citation_by_url("/documents/f1?page=1&chunk=c1")
    assert citation and citation["file_name"] == "a2.pdf"
    assert db.get_citation_by_url("/documents/missing?page=1&chunk=x") is None


def test_notebook_service_ingest_and_ask_with_monkeypatched_deps(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeDB:
        def __init__(self, _: Path) -> None:
            self.upserted = None
            self.saved = None

        def upsert_document(self, record: DocumentRecord) -> None:
            self.upserted = record

        def save_answer(self, question: str, answer: Answer) -> int:
            self.saved = (question, answer.answer)
            return 1

    class FakeFiles:
        def __init__(self, base_dir: Path) -> None:
            base_dir.mkdir(parents=True, exist_ok=True)

        def save_upload(self, source_path: Path, target_name: str) -> Path:
            target = tmp_path / target_name
            target.write_bytes(source_path.read_bytes())
            return target

    class FakePipeline:
        def __init__(self, **_: object) -> None:
            self.last_filters = None

        def ingest(self, file_path: Path) -> DocumentRecord:
            return DocumentRecord("docx", file_path.name, "pdf", str(file_path))

        def ask(self, question: str, filters: dict | None = None) -> Answer:
            self.last_filters = filters
            return Answer(answer=f"resp:{question}", citations=[])

    monkeypatch.setattr("rag.runtime.AppDatabase", FakeDB)
    monkeypatch.setattr("rag.runtime.FileStorage", FakeFiles)
    monkeypatch.setattr("rag.runtime.RagPipeline", FakePipeline)

    cfg = AppConfig(
        root_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        files_dir=tmp_path / "storage" / "files",
        chroma_dir=tmp_path / "storage" / "chroma",
        db_path=tmp_path / "storage" / "app.db",
    )
    service = NotebookService(cfg)
    record = service.ingest_uploaded_file("x.pdf", b"content")
    assert record.file_name == "x.pdf"
    assert service.db.upserted.file_id == "docx"

    answer = service.ask("oi", file_id="docx")
    assert answer.answer == "resp:oi"
    assert service.pipeline.last_filters == {"file_id": "docx"}


def test_runtime_collection_name_from_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _collection_name_from_env() == "chunks_fake_embedding_v1"

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    assert _collection_name_from_env() == "chunks_text_embedding_3_small"
