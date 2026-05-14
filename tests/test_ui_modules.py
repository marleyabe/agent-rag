from __future__ import annotations

from models import Citation, DocumentRecord
from ui.chat import resolve_filter_file_id
from ui.citations import citation_title, citation_viewer_link
from ui.document_viewer import build_pdf_open_url, get_viewer_payload, pdf_anchor_download_name
from ui.upload import ingest_upload


def test_chat_filter_resolution() -> None:
    assert resolve_filter_file_id(False, None) is None
    record = DocumentRecord("f1", "a.pdf", "pdf", "/tmp/a.pdf")
    assert resolve_filter_file_id(True, record) == "f1"


def test_citation_title_and_viewer_link() -> None:
    c = Citation(
        file_name="a.pdf",
        page_number=2,
        paragraph_number=None,
        start_line=1,
        end_line=3,
        source_text="x",
        url="/documents/f1?page=2&chunk=c1",
    )
    title = citation_title(c, 1)
    assert title == "[1] pag. 2 · linhas 1-3"
    assert citation_viewer_link(c) == "?doc=f1&chunk=c1&page=2"


def test_document_viewer_payload_and_pdf_name(tmp_path) -> None:
    class FakeDB:
        def get_citation_by_url(self, url: str):
            return {"url": url}

        def get_document(self, file_id: str):
            return {"file_id": file_id}

    payload = get_viewer_payload(FakeDB(), file_id="f1", chunk="c1", page="2")
    assert payload["citation"]["url"] == "/documents/f1?page=2&chunk=c1"
    assert payload["document"]["file_id"] == "f1"
    assert pdf_anchor_download_name("a.pdf", 2) == "a_page_2.pdf"
    assert pdf_anchor_download_name("a.pdf", None) == "a.pdf"
    assert build_pdf_open_url("/tmp/a.pdf", 3) == "/tmp/a.pdf#page=3"
    assert build_pdf_open_url("/tmp/a.pdf", None) == "/tmp/a.pdf"


def test_upload_ingest_delegates_to_service() -> None:
    class FakeService:
        def ingest_uploaded_file(self, file_name: str, content: bytes) -> DocumentRecord:
            assert file_name == "x.pdf"
            assert content == b"123"
            return DocumentRecord("f1", "x.pdf", "pdf", "/tmp/x.pdf")

    out = ingest_upload(FakeService(), "x.pdf", b"123")
    assert out.file_id == "f1"
