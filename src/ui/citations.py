from __future__ import annotations

from models import Citation
from ui.viewer import build_viewer_link, parse_citation_url


def citation_title(citation: Citation, idx: int) -> str:
    if citation.page_number is not None:
        return f"[{idx}] pag. {citation.page_number} · linhas {citation.start_line or '-'}-{citation.end_line or '-'}"
    return (
        f"[{idx}] par. {citation.paragraph_number or '-'} · "
        f"linhas {citation.start_line or '-'}-{citation.end_line or '-'}"
    )


def citation_viewer_link(citation: Citation) -> str:
    info = parse_citation_url(citation.url)
    return build_viewer_link(
        file_id=info["file_id"],
        chunk=info["chunk"],
        page=info["page"],
        paragraph=info["paragraph"],
    )
