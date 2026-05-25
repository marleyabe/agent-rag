# RAG Notebook MVP

MVP estilo NotebookLM com upload de PDF/DOCX, indexacao RAG, perguntas sobre documentos e respostas com citacoes rastreaveis.

## Funcionalidades

- Upload de PDF ou DOCX pela interface Streamlit.
- Indexacao em ChromaDB com metadados de arquivo, pagina/paragrafo e linhas do trecho.
- Perguntas factuais sobre o documento indexado.
- Perguntas genericas de visao geral, como resumo e assunto principal.
- Filtro opcional pelo ultimo documento enviado.
- Citacoes clicaveis que apontam para o trecho usado na resposta.
- Fallback extrativo local quando o LLM externo nao retorna evidencia, mas o retriever encontrou contexto relevante.

## Arquitetura

- `app.py`: interface Streamlit e estado da sessao.
- `src/ingestion`: extracao de PDF/DOCX e chunking.
- `src/vectorstore`: stores fake e ChromaDB.
- `src/rag`: embeddings, LLM, retriever, gerador de resposta e pipeline.
- `src/storage`: arquivos enviados e metadados/historico em SQLite.
- `src/citations`: formatacao de citacoes e links internos.
- `src/ui`: modulos de upload, chat, citacoes e visualizacao.
- `tests/`: testes unitarios e de fluxo deterministico.

## Tecnologias Usadas

- Python 3.11: linguagem principal do projeto.
- Streamlit: interface web para upload, chat e visualizacao das citacoes.
- ChromaDB: banco vetorial persistente usado para armazenar e buscar chunks por embeddings.
- OpenAI SDK: cliente para chamadas de embeddings e LLM quando `OPENAI_API_KEY` esta configurada.
- PyMuPDF (`fitz`): extracao de texto de arquivos PDF por pagina.
- python-docx: extracao de texto de arquivos DOCX por paragrafo.
- SQLite: persistencia local de documentos indexados, historico de perguntas e citacoes.
- python-dotenv: carregamento local das variaveis definidas em `.env`.
- uv: gerenciamento de ambiente, dependencias e execucao de comandos do projeto.
- pytest: suite de testes automatizados.
- pytest-cov: medicao de cobertura com gate minimo de 95%.
- Ruff: lint para manter qualidade e consistencia do codigo.
- Black: formatador de codigo Python configurado no projeto.

Componentes internos importantes:

- `DocumentLoader`: converte PDF/DOCX em secoes com metadados.
- `Chunker`: quebra secoes em chunks com linhas e sobreposicao.
- `Retriever`: combina busca vetorial e busca lexical para encontrar contexto relevante.
- `RagPipeline`: orquestra ingestao, recuperacao e geracao de respostas.
- `AnswerGenerator`: chama o LLM e monta resposta com citacoes.
- `CitationFormatter`: transforma metadados dos chunks em links rastreaveis.
- `FakeEmbeddingModel` e `FakeLLM`: implementacoes deterministicas para testes e desenvolvimento sem rede.

## Instalacao

Com `uv`:

```bash
uv sync --group dev
```

## Rodar App

```bash
uv run streamlit run app.py
```

Fluxo recomendado:

1. Abra o app.
2. Envie um PDF ou DOCX.
3. Clique em `Indexar documento`.
4. Pergunte no chat.
5. Use as citacoes para abrir o trecho relacionado.

A interface tambem tenta indexar automaticamente o arquivo selecionado antes de responder, caso ele ainda nao tenha sido indexado na sessao atual.

## Configuracao de Modelos

Configure `.env` com base em `.env.example` para usar OpenAI:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Sem `OPENAI_API_KEY`, o projeto usa modelos fake deterministico para desenvolvimento e testes locais.

## Pipeline RAG

1. Upload do arquivo.
2. Salvamento em `storage/files/`.
3. Extracao de secoes: PDF por pagina, DOCX por paragrafo.
4. Chunking com metadados de linha.
5. Geracao de embeddings.
6. Upsert no ChromaDB.
7. Recuperacao hibrida:
   - busca vetorial;
   - varredura lexical dos chunks do documento filtrado;
   - rerank por score hibrido.
8. Selecao de contexto.
9. Geracao da resposta.
10. Formatacao de citacoes.

## Perguntas Genericas

O pipeline trata perguntas de visao geral por uma rota especifica, porque elas nao apontam para um trecho unico do documento. Exemplos:

- `Sobre o que o documento fala?`
- `Qual e o tema principal do documento?`
- `Faca um resumo do documento.`
- `Quais sao os principais assuntos abordados?`
- `Qual e a estrutura geral do documento?`
- `Quais topicos aparecem no documento?`

Para esse tipo de pergunta, o pipeline prioriza capa, primeiras paginas, sumario e trechos de estrutura do documento.

## Perguntas Factuais

Perguntas factuais usam recuperacao hibrida nos chunks indexados. Exemplos para uma Constituicao:

- `Quais sao os direitos fundamentais?`
- `O que diz o artigo 5o?`
- `Quais sao os direitos sociais?`
- `Quais sao os principios fundamentais?`
- `Como a Constituicao organiza os poderes?`

## Citacoes

Cada chunk carrega:

- `file_id`
- `file_name`
- `file_type`
- `page_number` ou `paragraph_number`
- `start_line`
- `end_line`
- `source_text`
- `chunk_id`

O formatter cria links previsiveis:

- PDF: `/documents/{file_id}?page={page_number}&chunk={chunk_id}`
- DOCX: `/documents/{file_id}?paragraph={paragraph_number}&chunk={chunk_id}`

Na UI, esses links sao renderizados como parametros internos da propria pagina Streamlit.

## Testes

Rodar a suite completa:

```bash
uv run pytest
```

Rodar sem cobertura, util para ciclo rapido:

```bash
uv run pytest --no-cov
```

O `pytest` padrao inclui gate de cobertura minima de 95%.

Tambem ha cobertura do fluxo de UI com o test harness do Streamlit, incluindo upload, indexacao do `CF.pdf` e pergunta pelo chat.

## Validacao Manual Sugerida

Depois de iniciar o app:

1. Selecione `CF.pdf`.
2. Clique em `Indexar documento`.
3. Pergunte `Quais sao os direitos fundamentais?`.
4. Confirme que a resposta nao e `Nao encontrei essa informacao nos documentos enviados.`.
5. Confirme que existem citacoes abaixo da resposta.

## Limitações Conhecidas

- Linhas em PDF sao aproximadas.
- PDFs escaneados exigem OCR para boa extracao.
- A qualidade da resposta depende da qualidade do texto extraido.
- Modelos externos podem ser mais conservadores que o retriever; nesses casos o sistema usa fallback extrativo quando ha contexto recuperado.

## Proximos Passos

- OCR para PDFs escaneados.
- Visualizacao com highlight no trecho original.
- Resumos persistidos por documento.
- Avaliacao automatica de respostas.
- Autenticacao e suporte multiusuario.
