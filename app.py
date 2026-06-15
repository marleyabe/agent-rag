from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from rag.runtime import NotebookService  # noqa: E402
from ui.chat import ask_question, resolve_filter_file_id  # noqa: E402
from ui.citations import citation_title, citation_viewer_link  # noqa: E402
from ui.document_viewer import build_pdf_open_url, get_viewer_payload  # noqa: E402
from ui.upload import ingest_upload  # noqa: E402

SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_BATCH_SIZE",
    "RAG_MIN_RETRIEVAL_SCORE",
)


def upload_signature(file_name: str, content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()[:12]
    return f"{file_name}:{digest}"


def load_streamlit_secrets_into_env() -> None:
    for key in SECRET_ENV_KEYS:
        if os.getenv(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value is not None:
            os.environ[key] = str(value)


def main() -> None:
    st.set_page_config(page_title="RAG Notebook MVP", layout="wide")
    st.title("RAG Notebook MVP")
    load_streamlit_secrets_into_env()

    if "service" not in st.session_state:
        st.session_state.service = NotebookService(AppConfig())
    if "last_record" not in st.session_state:
        st.session_state.last_record = None
    if "last_upload_signature" not in st.session_state:
        st.session_state.last_upload_signature = None
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
            uploaded_content = uploaded.getvalue()
            signature = upload_signature(uploaded.name, uploaded_content)
            progress_text = st.empty()
            progress_bar = st.progress(0)

            def _on_progress(value: float, message: str) -> None:
                bounded = max(0.0, min(1.0, float(value)))
                progress_bar.progress(int(bounded * 100))
                progress_text.info(message)

            try:
                record = ingest_upload(
                    service,
                    uploaded.name,
                    uploaded_content,
                    progress_callback=_on_progress,
                )
            except Exception as exc:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"Nao foi possivel indexar o documento: {exc}")
            else:
                progress_bar.empty()
                progress_text.empty()
                st.session_state.last_record = record
                st.session_state.last_upload_signature = signature
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
        if uploaded:
            uploaded_content = uploaded.getvalue()
            signature = upload_signature(uploaded.name, uploaded_content)
            if st.session_state.last_upload_signature != signature:
                try:
                    with st.spinner("Indexando documento selecionado antes de responder..."):
                        record = ingest_upload(service, uploaded.name, uploaded_content)
                except Exception as exc:
                    st.error(f"Nao foi possivel indexar o documento: {exc}")
                    return
                else:
                    st.session_state.last_record = record
                    st.session_state.last_upload_signature = signature

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        file_id = resolve_filter_file_id(only_last, st.session_state.last_record)
        try:
            answer = ask_question(service, question, file_id=file_id)
        except Exception as exc:
            st.error(f"Nao foi possivel responder agora: {exc}")
            return
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
