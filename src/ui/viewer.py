from __future__ import annotations

from urllib.parse import parse_qs


def parse_citation_url(url: str) -> dict[str, str]:
    parts = url.split("?", maxsplit=1)
    file_id = parts[0].split("/")[-1]
    query = parse_qs(parts[1] if len(parts) > 1 else "")
    return {
        "file_id": file_id,
        "page": query.get("page", [""])[0],
        "paragraph": query.get("paragraph", [""])[0],
        "chunk": query.get("chunk", [""])[0],
    }


def build_viewer_link(file_id: str, chunk: str, page: str = "", paragraph: str = "") -> str:
    params = [f"doc={file_id}", f"chunk={chunk}"]
    if page:
        params.append(f"page={page}")
    if paragraph:
        params.append(f"paragraph={paragraph}")
    return "?" + "&".join(params)


def build_citation_lookup_url(file_id: str, chunk: str, page: str = "", paragraph: str = "") -> str:
    query_parts: list[str] = []
    if page:
        query_parts.append(f"page={page}")
    if paragraph:
        query_parts.append(f"paragraph={paragraph}")
    query_parts.append(f"chunk={chunk}")
    return f"/documents/{file_id}?" + "&".join(query_parts)

