# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```bash
docker-compose up -d    # Start Qdrant (port 6333)
uv sync                 # Install Python dependencies
```

**Run the application:**
```bash
uv run main.py --chat                                              # Start RAG chat
uv run main.py --run-etl --path /path/to/file_or_folder           # Ingest any supported file(s)
uv run main.py --run-etl --erase --path /path/to/file_or_folder   # Erase DB then ingest
uv run main.py --run-etl --recursive-chunking --path ...          # Use recursive instead of semantic chunking
uv run main.py --check-dbs                                         # Check collection stats
uv run main.py --embed-model BAAI/bge-m3 --chat                   # Use a different embedding model (fastembed or HuggingFace)
```

**Tests:**
```bash
# Unit tests (no infrastructure needed)
uv run pytest tests/agent/

# RAG quality tests (requires running infrastructure + populated DB)
# These generate answers first, then judge them with an LLM
uv run pytest tests/rag/ --model llama3.1:8b --questions tests/rag/questions/questions.json
uv run pytest tests/rag/ --model llama3.1:8b --regen    # Force re-generation of answers
uv run pytest tests/rag/ --collection-name MyCollection  # Use a specific collection
```

## Architecture

### Infrastructure
- **Qdrant** (localhost:6333): Vector database using hybrid search (dense + sparse vectors, 384-dim, DOT distance)
- **Ollama** (localhost:11434): Local LLM runtime (default model: `llama3.1:8b`)

### Module Overview

**`etl/`** — Data ingestion pipeline (ETL pattern)
- `BaseEtl` → abstract base with `extract()`, `transform()`, `load()` state machine; `OUTPUT_FOLDER` class attr controls where converted Markdown is written
- `GeneralEtl` → **primary ETL** for any file type; converts to Markdown in-process, splits by headers, extracts tables, cleans text, semantic/recursive chunks (`chunk_size=768`, `chunk_overlap=200`), embeds; output goes to `data/general/`
- `DroughtEtl` → legacy ETL for climate PDFs only; reads pre-converted Markdown from `data/drough/`; includes LLM-based metadata extraction (years, locations, entities)
- `converters.py` → registry of per-extension converters (`@register_converter`); called by `BaseEtl.extract()`
- `loaders.py` → loader registry; unregistered extensions fall back to `_insert_by_chunks()` (the default for all `GeneralEtl` file types)
- `table_extractor.py` → `TableProcessor` / `MarkdownTableExtractor`; extracts and summarises Markdown tables before text splitting

**Supported file types (`GeneralEtl`):**

| Extension | Processing |
|---|---|
| `.pdf`, `.docx`, `.pptx` | Docling `DocumentConverter` → Markdown → header-split → semantic/recursive chunk → embed |
| `.md`, `.txt` | Copied to `data/general/` → same Markdown pipeline as above |
| `.csv`, `.xlsx` | **Row-per-document**: each row → pipe-delimited text + all column values as typed metadata; embedded directly, no chunking |

CSV/XLSX metadata example: `{"source": "games_2025", "file_type": "csv", "name": "Split Fiction", "category": "Co-op Adventure", "review": 9.4, "row_index": 13, "text": "name: Split Fiction | ...", ...}` — column names become metadata keys dynamically from headers, enabling Qdrant numeric/keyword filtering per dataset.

**`rag/agents/`** — Agentic RAG using LangGraph
- `AgenticRAG(database_service, embedding_service, model_name, use_doc_summaries=False)` → main entry point
- `AgenticRAG.chat(question)` → returns `{response, sources, rewritten_query, extracted_data}`
- `use_doc_summaries` flag (default `False`): when `True`, enables two-stage retrieval (Stage-0 doc-summary search before chunk search); set to `False` for baseline/comparison runs
- Graph nodes in `rag/agents/nodes/`: `general_nodes.py` (Router, General), `rag_nodes.py` (QueryRewriter, Extractor, ResearchWorker, RetrievalGrader, HallucinationGrader, Synthesizer, Error)
- `build_graph(..., use_doc_summaries=False)` in `graph.py` passes flag to `RagNodes`
- State is `AgentState` in `rag/agents/state.py`

**`metadata_extractor/`** — Standalone LangGraph sub-graph
- Used only by `DroughtEtl` to extract `years`, `locations`, `entities`, `has_numerical_data` from chunks via LLM
- `build_extractor_graph()` → compiled graph; `extract_metadata(text)` → convenience function

**`database/`** — Storage abstraction
- `BaseDbRepository` (ABC) → defines the interface: `connect`, `search`, `insert`, `delete`, `get_count`, etc.
- `QdrantDbRepository` → production implementation with hybrid search (dense + sparse)
- `MyDocument` → internal document model with `id`, `text`, `embedding`, `sparse_embedding`, `metadata`

**`text_embedding/`** — In-process embedding module (no HTTP/Docker required)
- `TextEmbeddingService(library="fastembed", dense_model=None, sparse_model=...)` → main entry point
- `get_embedding_with_uuid(data, chunk_size=None)` → returns `List[EmbeddingResponse]` with dense + sparse vectors
- `get_embedding_dim()` → returns output dimensionality of the current dense model (used by `main.py` to auto-configure Qdrant)
- `set_library("fastembed"|"sentence_transformers")` / `set_model(name)` for runtime switching
- **Auto-detection**: when `library="fastembed"` (default) but the requested model is not in fastembed's supported list, automatically falls back to `sentence_transformers` — allows any HuggingFace model via `--embed-model`
- Dense libraries: `fastembed` (default: `BAAI/bge-small-en-v1.5`, 384-dim) and `sentence_transformers` (default: `all-MiniLM-L6-v2`)
- Sparse: always `fastembed` SPLADE (`prithivida/Splade_PP_en_v1`)
- `text_embedding_api/` still exists as a standalone Dockerized FastAPI server but is no longer used by the main codebase
- **`VECTOR_DB_VECTOR_SIZE` env var is no longer used** — vector size is auto-derived from the loaded model via `get_embedding_dim()`; must re-ingest with `--erase` when switching models

**`semantic_chunking/`** — Similarity-based text splitting
- `SimilarSentenceSplitter` → splits text by grouping sentences with high cosine similarity
- Used by both `GeneralEtl` and `DroughtEtl` when `use_semantic=True`

**`utils/Utils.py`** — Shared utilities
- `Utils.convert_to_md(path, output_folder)` → converts any Docling-supported file to Markdown
- `Utils.convert_pdf_to_md(path, output_folder)` → thin wrapper around `convert_to_md` (kept for compatibility)

**`tests/`**
- `tests/agent/` — Unit tests for RAG agent nodes with mocked dependencies
- `tests/rag/` — End-to-end quality tests: `generate_answers.py` runs the full RAG pipeline and saves results to `tests/rag/results/`; `test_rag_custom.py` reads saved answers and judges them with an LLM `Judge`

### Key Data Flow

**ETL (GeneralEtl):** Any file → `BaseEtl.extract()` (converter registry) → Markdown in `data/general/` → `GeneralEtl.transform()` (extract tables, split by H1–H4 headers, clean, semantic/recursive chunk, embed) → Qdrant

**ETL metadata per chunk (documents):** `source` (filename stem), `file_type` (no leading dot, e.g. `pdf`), `header_path`, `headers`, `is_table`, `chunk_index`

**ETL metadata per row (CSV/XLSX):** `source`, `file_type`, `is_table`, `row_index`, `text`, + one key per column with its typed value (int/float/str)

**`_clean_text()` in `GeneralEtl`** strips markdown syntax before embedding: `**bold**`/`__bold__` → plain text, `*italic*` → plain text, `` `code` `` → plain text, `[text](url)` → `text`, `> blockquote` → text, `---` horizontal rules removed. Underscore-italic (`_text_`) is intentionally left alone to avoid corrupting filenames and identifiers.

**RAG query:** User question → Router → (General path OR RAG path: QueryRewriter → Extractor → ResearchWorker → RetrievalGrader → Synthesizer → HallucinationGrader) → Answer

### Notes
- `constants.py` defines collection names (`COLLECTION_NAME_DROUGH = 'drough'`)
- `--vector-db` flag exists in main.py but only `qdrant` is actively used
- Tests in `tests/rag/` require infrastructure running and a populated Qdrant collection
- `DroughtEtl` is retained for the climate dataset but `GeneralEtl` is the active default
- Reranker in `rag_nodes.py` (`retrieval_grader_agent`) keeps top **20** docs after reranking (increased from 10); improves coverage for listing/enumeration questions
- Re-indexing required when changing chunk parameters: `uv run main.py --run-etl --erase --path <folder>`

### Document Metadata Schema

Each ingested file now produces **one synthetic doc-summary point** (in addition to regular chunk points). The summary point's `text` field contains both the LLM summary sentence and the full header outline so the embedding captures both signals.

**Fields on the doc-summary point** (`is_doc_summary=True`):
- `is_doc_summary: bool` — `True` only on the synthetic summary point (one per file)
- `doc_summary: str` — LLM one-sentence description of the document
- `doc_headers_outline: str` — all H1–H4 headers joined by ` > `

**Fields propagated to ALL chunk points** (`is_doc_summary=False`):
- `temporal_scope_start: int | null` — first year the document covers (null if timeless/unknown)
- `temporal_scope_end: int | null` — last year the document covers (null if timeless/unknown)
- `is_doc_summary: bool` — always `False` on regular chunks

**Two-stage retrieval** in `research_worker` (`rag_nodes.py`) — active only when `use_doc_summaries=True`:
1. **Stage 0** — search only `is_doc_summary=True` points to find the top-5 most relevant source files
2. **Strategy 0a** — if strict metadata filters (year/location) also exist, prepend `{source: relevant_sources, is_doc_summary: False, **strict_filter}` (confidence=0.95)
3. **Strategy 0b** — always prepend `{source: relevant_sources, is_doc_summary: False}` (confidence=0.85) when sources found
4. All strategies include `is_doc_summary=False` regardless of mode to keep synthetic summary points out of chunk results

This eliminates cross-year and cross-domain contamination without changing the collection schema or query interface.

**`use_doc_summaries` flag** — controls two-stage retrieval on/off:
- Default is `False` (baseline mode) to allow A/B comparison against the old single-stage retrieval
- Enable with `AgenticRAG(..., use_doc_summaries=True)` or via test `--use-doc-summaries` flag

### RAG Prompt Design
Key rules currently in effect in `rag/agents/prompts.py` (Synthesizer node):
- **COMPLETENESS RULE**: For enumeration/listing questions, compile partial lists from *all* retrieved sources rather than stopping at the first match
- **TEMPORAL ACCURACY RULE**: Only include facts that apply to the year explicitly asked; do not mix data from other years
- **GROUNDING RULE**: Refuse to answer only when *none* of the sources are topically relevant — do not refuse merely because the answer is incomplete

Context compressor rule (also in `prompts.py`): when uncertain whether a sentence is relevant, return the document unchanged rather than truncating (prevents injecting noise into the Synthesizer).
