from __future__ import annotations

from ui.viewer import build_citation_lookup_url, build_viewer_link, parse_citation_url


def test_parse_citation_url_roundtrip_pdf() -> None:
    url = "/documents/doc123?page=2&chunk=doc123_p2_c0"
    parsed = parse_citation_url(url)
    assert parsed == {"file_id": "doc123", "page": "2", "paragraph": "", "chunk": "doc123_p2_c0"}
    assert build_citation_lookup_url("doc123", "doc123_p2_c0", page="2") == url


def test_build_viewer_link_omits_empty_params() -> None:
    assert build_viewer_link("doc1", "c1") == "?doc=doc1&chunk=c1"
    assert build_viewer_link("doc1", "c1", paragraph="3") == "?doc=doc1&chunk=c1&paragraph=3"

