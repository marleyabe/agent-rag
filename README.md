# RAG Notebook MVP

MVP estilo NotebookLM com upload de PDF/DOCX, indexacao RAG, perguntas sobre documentos e respostas com citacoes rastreaveis.

## Integrantes

- Marley Abe Silva
- Maycon Moriy Abe Machado

## Funcionalidades

- Upload de PDF ou DOCX pela interface Streamlit.
- Indexacao em ChromaDB com metadados de arquivo, pagina/paragrafo e linhas do trecho.
- Perguntas factuais sobre o documento indexado.
- Perguntas genericas de visao geral, como resumo e assunto principal.
- Filtro opcional pelo ultimo documento enviado.
- Citacoes clicaveis que apontam para o trecho usado na resposta.
- Leitura de secrets do Streamlit Cloud para configuracao de modelos em deploy.
- Tratamento de erros de upload, indexacao e resposta direto na interface.
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
- `cartilha_ppsi.pdf`: documento curto sugerido para demonstracao do prototipo.

## Tecnologias Usadas

- Python 3.11: linguagem principal do projeto.
- Streamlit: interface web para upload, chat e visualizacao das citacoes.
- ChromaDB: banco vetorial persistente usado para armazenar e buscar chunks por embeddings.
- OpenAI SDK: cliente para chamadas de embeddings e LLM quando `OPENAI_API_KEY` esta configurada.
- PyMuPDF (`fitz`): extracao de texto de arquivos PDF por pagina.
- python-docx: extracao de texto de arquivos DOCX por paragrafo.
- SQLite: persistencia local de documentos indexados, historico de perguntas e citacoes.
- python-dotenv: carregamento local das variaveis definidas em `.env`.
- Playwright: validacao end-to-end manual do fluxo no navegador.
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

No Streamlit Cloud, configure os mesmos valores em `Settings > Secrets`:

```toml
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

O app tambem aceita `OPENAI_EMBEDDING_BATCH_SIZE` e `RAG_MIN_RETRIEVAL_SCORE` como variaveis opcionais.

## Deploy no Streamlit Cloud

Configuracao recomendada para apresentacao:

1. Selecione o repositorio no Streamlit Community Cloud.
2. Use `app.py` como arquivo principal.
3. Configure Python 3.11.
4. Adicione os secrets da OpenAI no painel do Streamlit.
5. Faca o upload de um PDF/DOCX pela interface do app.

Observacoes importantes:

- O app salva uploads, SQLite e ChromaDB em `storage/` dentro do container.
- Esse armazenamento e suficiente para demonstracao, mas pode ser perdido em restart ou redeploy.
- Se o ChromaDB persistente falhar no ambiente do Streamlit Cloud, o app usa um store vetorial em memoria para manter a demo funcionando.
- Se o log mostrar `ClientDisconnect` durante upload, tente reenviar o arquivo; esse erro vem da conexao de upload do Streamlit.
- Nao suba `.env` para o repositorio; use secrets no Streamlit Cloud.
- Para demo em sala, mantenha tambem o app local como plano B.

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

Quando a OpenAI esta configurada, o gerador primeiro tenta extrair fatos objetivos do contexto recuperado e depois monta a resposta citada. Se a etapa de extracao for conservadora demais, o modelo ainda recebe o contexto recuperado antes de o sistema cair no fallback local.

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

Perguntas factuais usam recuperacao hibrida nos chunks indexados. Exemplos para a `cartilha_ppsi.pdf`:

- `Qual e o objetivo do PPSI?`
- `Quais sao as areas de atuacao do PPSI?`
- `O que faz a area de tecnologia?`
- `Qual e a missao do CISC Gov.br?`
- `Quais sao os objetivos do Centro de Excelencia?`
- `Quais sao as etapas para implementacao do Framework?`

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

Tambem ha cobertura do fluxo de UI com o test harness do Streamlit.

Validacao end-to-end manual usada neste projeto:

```bash
uv run streamlit run app.py --server.headless true --server.port 8504
uv run --with playwright python tests/e2e_streamlit_llm.py
```

O roteiro Playwright valida:

- abertura do app;
- upload e indexacao da `cartilha_ppsi.pdf`;
- pergunta no chat;
- resposta com LLM real quando `OPENAI_API_KEY` esta configurada;
- renderizacao de citacoes;
- abertura da visualizacao da citacao.

## Validacao Manual Sugerida

Depois de iniciar o app:

1. Selecione `cartilha_ppsi.pdf`.
2. Clique em `Indexar documento`.
3. Pergunte `Qual e o objetivo do PPSI?`.
4. Confirme que a resposta nao e `Nao encontrei essa informacao nos documentos enviados.`.
5. Confirme que existem citacoes abaixo da resposta.
6. Clique em uma citacao e confirme que a tela mostra arquivo, pagina, linhas e trecho citado.

Documento recomendado para demonstracao:

- `cartilha_ppsi.pdf`, uma cartilha oficial curta do Programa de Privacidade e Seguranca da Informacao.
- Por ser pequena e ter texto extraivel, costuma indexar rapido e gerar respostas objetivas.

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
