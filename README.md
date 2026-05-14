# RAG Notebook MVP

MVP estilo NotebookLM com upload de PDF/DOCX, indexação RAG e respostas com citações rastreáveis.

## Arquitetura

- `app.py`: interface Streamlit.
- `src/ingestion`: extração e chunking.
- `src/vectorstore`: implementação fake e ChromaDB.
- `src/rag`: `embedding_core`, `llm_core`, retriever, gerador de resposta e pipeline.
- `src/storage`: armazenamento de arquivos e metadados/histórico em SQLite.
- `src/citations`: formatação de citações e links internos.
- `src/ui`: módulos de UI (`upload`, `chat`, `citations`, `document_viewer`, `viewer`).
- `tests/`: testes unitários determinísticos.

## Instalação

Com `uv`:

```bash
uv sync --group dev
```

## Rodar testes

```bash
uv run python -m pytest
```

O `pytest` já inclui gate de cobertura mínima de 95%.

## Rodar app

```bash
uv run streamlit run app.py
```

Configure `.env` (base em `.env.example`) para usar Gemini:

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (padrão: `models/gemini-3-flash-preview`)
- `GEMINI_EMBEDDING_MODEL` (padrão: `models/gemini-embedding-001`)

## Pipeline RAG

1. Upload de arquivo.
2. Salvamento em `storage/files/`.
3. Extração de seções (PDF por página, DOCX por parágrafo).
4. Chunking com metadados de linha.
5. Embeddings.
6. Indexação no ChromaDB.
7. Recuperação dos chunks mais relevantes.
8. Geração da resposta e citações.

## Como as citações são geradas

Cada chunk carrega `file_id`, `file_name`, `page_number`/`paragraph_number`, `start_line`, `end_line`, `source_text` e `chunk_id`.  
O formatter cria links previsíveis:

- PDF: `/documents/{file_id}?page={page_number}&chunk={chunk_id}`
- DOCX: `/documents/{file_id}?paragraph={paragraph_number}&chunk={chunk_id}`

## Limitações conhecidas

- Linhas em PDF são aproximadas.
- PDFs escaneados podem exigir OCR.
- Qualidade das citações depende da extração de texto.

## Próximos passos

- OCR para PDF escaneado.
- Visualização com highlight no trecho original.
- Autenticação.
- Suporte multiusuário.
- Re-ranking de contexto.
- Avaliação automática de respostas.
