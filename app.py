from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig
from rag.runtime import NotebookService
from ui.chat import ask_question, resolve_filter_file_id
from ui.citations import citation_title, citation_viewer_link
from ui.document_viewer import build_pdf_open_url, get_viewer_payload
from ui.upload import ingest_upload
from ui.viewer import parse_citation_url


def main() -> None:
    st.set_page_config(page_title="RAG Notebook MVP", layout="wide")
    st.title("RAG Notebook MVP")

    if "service" not in st.session_state:
        st.session_state.service = NotebookService(AppConfig())
    if "last_record" not in st.session_state:
        st.session_state.last_record = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    service: NotebookService = st.session_state.service
    query = st.query_params
    doc_file_id = query.get("doc")
    doc_chunk = query.get("chunk")
    doc_page = query.get("page")
    doc_paragraph = query.get("paragraph")

    if doc_file_id and doc_chunk:
        payload = get_viewer_payload(
            db=service.db,
            file_id=doc_file_id,
            chunk=doc_chunk,
            page=doc_page or "",
            paragraph=doc_paragraph or "",
        )
        citation = payload["citation"]
        document = payload["document"]
        st.subheader("Visualizacao de citacao")
        if citation:
            st.write(f"Arquivo: {citation['file_name']}")
            st.write(
                f"Pagina: {citation['page_number'] or '-'} | "
                f"Paragrafo: {citation['paragraph_number'] or '-'} | "
                f"Linhas: {citation['start_line'] or '-'}-{citation['end_line'] or '-'}"
            )
            st.code(citation["source_text"])
        if document:
            if citation and citation.get("page_number") and document.get("file_type") == "pdf":
                st.markdown(
                    f"[Abrir PDF na pagina citada]({build_pdf_open_url(document['file_path'], citation['page_number'])})"
                )
        st.divider()

    with st.sidebar:
        st.subheader("Upload")
        uploaded = st.file_uploader("PDF ou DOCX", type=["pdf", "docx"])
        if uploaded and st.button("Indexar documento"):
            record = ingest_upload(service, uploaded.name, uploaded.getvalue())
            st.session_state.last_record = record
            st.success(f"Documento indexado: {record.file_name}")

    only_last = st.checkbox("Filtrar pelo ultimo documento enviado", value=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("citations"):
                inline = " · ".join(
                    [f"<a href=\"{item['viewer_url']}\">{item['label']}</a>" for item in message["citations"]]
                )
                st.markdown(f"<div style='font-size:12px'>{inline}</div>", unsafe_allow_html=True)

    question = st.chat_input("Pergunte sobre os documentos")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        file_id = resolve_filter_file_id(only_last, st.session_state.last_record)
        answer = ask_question(service, question, file_id=file_id)
        rendered_citations = []
        with st.chat_message("assistant"):
            st.write(answer.answer)
            inline_parts = []
            for idx, citation in enumerate(answer.citations, start=1):
                label = citation_title(citation, idx)
                viewer_url = citation_viewer_link(citation)
                inline_parts.append(f"<a href=\"{viewer_url}\">{label}</a>")
                rendered_citations.append(
                    {
                        "label": label,
                        "viewer_url": viewer_url,
                    }
                )
            if inline_parts:
                st.markdown(
                    f"<div style='font-size:12px'>{' · '.join(inline_parts)}</div>",
                    unsafe_allow_html=True,
                )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer.answer,
                "citations": rendered_citations,
            }
        )
        st.divider()


if __name__ == "__main__":
    main()
