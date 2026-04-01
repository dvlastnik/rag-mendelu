# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Docker (primary workflow):**
```bash
# Ollama must be running on the host first (GPU used natively on any platform):
#   macOS/Windows: open Ollama desktop app   |   Linux: ollama serve
# Then pull your model:  ollama pull ministral-3:8b
docker compose up -d                         # Start Qdrant only
docker compose run --rm -it app              # Launch TUI wizard (no args)
docker compose run --rm -it app --chat       # Interactive RAG chat (skip TUI)
docker compose run --rm app --run-etl --path /data/input/   # Ingest from ./data/input/
docker compose run --rm app --run-etl --erase --path /data/input/
docker compose run --rm app --check-dbs
# Mount arbitrary host path for ETL (no copying needed):
docker compose run --rm -v /your/path:/data/input:ro app --run-etl --path /data/input/
```

**Local dev (no Docker, for tests and development):**
```bash
docker compose up -d qdrant   # Start only Qdrant (Ollama stays on host)
uv sync                       # Install Python dependencies
uv run main.py                # Launch interactive TUI (no args → TUI wizard)
uv run main.py --chat
uv run main.py --run-etl --path /path/to/file_or_folder
uv run main.py --run-etl --erase --path /path/to/file_or_folder
uv run main.py --run-etl --recursive-chunking --path ...
uv run main.py --check-dbs
uv run main.py --embed-model BAAI/bge-m3 --chat
```

**Tests:**
```bash
# Unit tests (no infrastructure needed)
uv run pytest tests/agent/

# RAG quality tests (requires running infrastructure + populated DB)
# These generate answers first, then judge them with an LLM
uv run pytest tests/rag/ --model ministral-3:8b --questions tests/rag/questions/questions.json
uv run pytest tests/rag/ --model ministral-3:8b --regen    # Force re-generation of answers
uv run pytest tests/rag/ --collection-name MyCollection  # Use a specific collection

# Run tests across all loaded Ollama models (size-sorted, smallest first)
./run_tests.sh                                                          # default collection=drough
./run_tests.sh --collection random --questions tests/rag/questions/questions_random.json
```

## Architecture

### Infrastructure
- **Qdrant** (container, port 6333): Vector database using hybrid search (dense + sparse vectors, 384-dim, DOT distance); data in `qdrant_data` volume
- **Ollama** (host process, port 11434): LLM runtime runs **locally on the host** (not in Docker) so GPU acceleration works natively on NVIDIA, AMD (ROCm), and Apple Silicon (Metal); app container reaches it via `host.docker.internal:11434`
- **DuckDB** (embedded in app): No separate service; file at `data/sql/tabular.duckdb` persisted in `duckdb_data` volume
- **app** (container, `profiles: [app]`): Python CLI; not auto-started by `docker compose up` — invoke with `docker compose run`

**Environment variables** (set in `docker-compose.yaml`, override via `.env`):
- `QDRANT_HOST` / `QDRANT_PORT` — Qdrant connection (default: `qdrant` / `6333`)
- `OLLAMA_HOST` — Ollama base URL (default in Docker: `http://host.docker.internal:11434`; local dev: `http://localhost:11434`)
- `OLLAMA_MODEL` — model to pull and use (default: `ministral-3:8b`)
- `COLLECTION_NAME` — Qdrant collection (default: `default_name`)
- `VECTOR_DB_DISTANCE` — distance metric (default: `DOT`)

### Module Overview

**`etl/`** — Data ingestion pipeline (ETL pattern)
- `BaseEtl` → abstract base with `extract()`, `transform()`, `load()` state machine; `OUTPUT_FOLDER` class attr controls where converted Markdown is written
- `GeneralEtl` → **primary ETL** for any file type; converts to Markdown in-process, splits by headers, extracts tables, cleans text, semantic/recursive chunks (`chunk_size=768`, `chunk_overlap=200`), embeds; output goes to `data/general/`
- `DroughtEtl` → legacy ETL for climate PDFs only; reads pre-converted Markdown from `data/drough/`; includes LLM-based metadata extraction (years, locations, entities)
- `converters.py` → registry of per-extension converters (`@register_converter`); called by `BaseEtl.extract()`
- `loaders.py` → loader registry; unregistered extensions fall back to `_insert_by_chunks()` (the default for all `GeneralEtl` file types)
- `table_extractor.py` → `TableProcessor` / `MarkdownTableExtractor`; extracts and summarises Markdown tables before text splitting

**Supported file types (`GeneralEtl`):**

| Extension | Qdrant | DuckDB | Notes |
|---|---|---|---|
| `.pdf`, `.docx`, `.pptx` | Text chunks + table rows | Extracted tables (one row per table row, columns as fields) | Docling `DocumentConverter` → Markdown → table extract → header-split → semantic/recursive chunk → embed |
| `.md`, `.txt` | Text chunks + table rows | Extracted tables | Copied to `data/general/` → same Markdown pipeline as above |
| `.csv`, `.xlsx` | **Row-per-document**: each row → pipe-delimited text + all column values as typed metadata; no chunking | Full file registered as DuckDB table for SQL aggregation | All columns stored as typed metadata in both stores |

CSV/XLSX metadata example: `{"source": "games_2025", "file_type": "csv", "name": "Split Fiction", "category": "Co-op Adventure", "review": 9.4, "row_index": 13, "text": "name: Split Fiction | ...", ...}` — column names become metadata keys dynamically from headers, enabling Qdrant numeric/keyword filtering per dataset.

**`rag/agents/`** — Agentic RAG using LangGraph
- `AgenticRAG(database_service, embedding_service, model_name)` → main entry point
- `AgenticRAG.chat(question)` → returns `{response, sources, rewritten_queries, sql_result, distilled_facts}`
- Graph nodes in `rag/agents/nodes/`: `general_nodes.py` (Router, General), `rag_nodes.py` (QueryPlanner, AnalyticalQuery, ScrollRetriever, ResearchWorker, RetrievalGrader, FactExtractor, Synthesizer, CompletenessChecker, HallucinationGrader, Error)
- State is `AgentState` in `rag/agents/state.py`

**`metadata_extractor/`** — Standalone LangGraph sub-graph
- Used only by `DroughtEtl` to extract `years`, `locations`, `entities`, `has_numerical_data` from chunks via LLM
- `build_extractor_graph()` → compiled graph; `extract_metadata(text)` → convenience function

**`database/`** — Storage abstraction
- `BaseDbRepository` (ABC) → defines the interface: `connect`, `search`, `insert`, `delete`, `get_count`, etc.
- `QdrantDbRepository` → production implementation with hybrid search (dense + sparse)
- `DuckDbRepository` → embedded DuckDB for analytical SQL queries on tabular data; key methods: `register_csv`, `register_xlsx`, `register_dataframe` (in-memory DataFrame), `run_select`, `get_compact_catalog`, `drop_table`
- `MyDocument` → internal document model with `id`, `text`, `embedding`, `sparse_embedding`, `metadata`

**`text_embedding/`** — In-process embedding module (no HTTP/Docker required)
- `TextEmbeddingService(library="fastembed", dense_model=None, sparse_model=...)` → main entry point
- `get_embedding_with_uuid(data, chunk_size=None)` → returns `List[EmbeddingResponse]` with dense + sparse vectors
- `get_embedding_dim()` → returns output dimensionality of the current dense model (used by `main.py` to auto-configure Qdrant)
- `set_library("fastembed"|"sentence_transformers")` / `set_model(name)` for runtime switching
- **Auto-detection**: when `library="fastembed"` (default) but the requested model is not in fastembed's supported list, automatically falls back to `sentence_transformers` — allows any HuggingFace model via `--embed-model`
- Dense libraries: `fastembed` (default: `BAAI/bge-small-en-v1.5`, 384-dim) and `sentence_transformers` (default: `all-MiniLM-L6-v2`)
- Sparse: always `fastembed` SPLADE (`prithivida/Splade_PP_en_v1`)
- **`VECTOR_DB_VECTOR_SIZE` env var is no longer used** — vector size is auto-derived from the loaded model via `get_embedding_dim()`; must re-ingest with `--erase` when switching models

**`semantic_chunking/`** — Similarity-based text splitting
- `SimilarSentenceSplitter` → splits text by grouping sentences with high cosine similarity
- Used by both `GeneralEtl` and `DroughtEtl` when `use_semantic=True`

**`utils/Utils.py`** — Shared utilities
- `Utils.convert_to_md(path, output_folder)` → converts any Docling-supported file to Markdown
- `Utils.convert_pdf_to_md(path, output_folder)` → thin wrapper around `convert_to_md` (kept for compatibility)

**`tui/`** — Interactive TUI wizard (InquirerPy)
- `TuiWizard(default_model, qdrant_host, qdrant_port)` — launched from `main.py` when no CLI flags are provided
- `run()` → returns `argparse.Namespace` consumed by the rest of `main()`
- `_get_ollama_models()` → `@staticmethod`; parses `ollama list` output
- `_get_qdrant_collections(host, port)` → `@staticmethod`; creates a temporary `QdrantClient` to fetch collection names; returns `[]` on any failure
- **Chat/Ask**: selects from existing collections (text fallback if none found)
- **ETL**: asks "New / Existing" first — new prompts for a name, existing shows a select list
- All CLI flags (`--chat`, `--run-etl`, etc.) bypass the TUI entirely

**`tests/`**
- `tests/agent/` — Unit tests for RAG agent nodes with mocked dependencies
- `tests/rag/` — End-to-end quality tests: `generate_answers.py` runs the full RAG pipeline and saves results to `tests/rag/results/`; `test_rag_custom.py` reads saved answers and judges them with an LLM `Judge`
- `tests/rag/questions/` — Question sets: `questions.json` (drough/climate), `questions_random.json` (30 questions over history_of_metal/lotr_lore/hussite_wars/games_2025)
- `run_tests.sh` — Bash script: discovers all loaded Ollama models via `ollama list`, sorts by size (smallest first), runs the full test suite per model, generates `evaluation_matrix.csv`

### Key Data Flow

**ETL (GeneralEtl):** Any file → `BaseEtl.extract()` (converter registry) → Markdown in `data/general/` → `GeneralEtl.transform()` (extract tables → Qdrant + DuckDB, split by H1–H4 headers, clean, semantic/recursive chunk, embed) → Qdrant. For CSV/XLSX: original file also registered in DuckDB via `register_csv`/`register_xlsx`. For PDF/DOCX/MD/TXT: extracted tables registered via `register_dataframe`.

**ETL metadata per chunk (documents):** `source` (filename stem), `file_type` (no leading dot, e.g. `pdf`), `header_path`, `headers`, `is_table`, `chunk_index`

**ETL metadata per row (CSV/XLSX):** `source`, `file_type`, `is_table`, `row_index`, `text`, + one key per column with its typed value (int/float/str)

**`_clean_text()` in `GeneralEtl`** strips markdown syntax before embedding: `**bold**`/`__bold__` → plain text, `*italic*` → plain text, `` `code` `` → plain text, `[text](url)` → `text`, `> blockquote` → text, `---` horizontal rules removed. Underscore-italic (`_text_`) is intentionally left alone to avoid corrupting filenames and identifiers.

**RAG query:** User question → Router → (General path OR RAG path: QueryPlanner → [SQL/HYBRID → AnalyticalQuery] / [VECTOR → ResearchWorker] / [SCROLL → ScrollRetriever] → RetrievalGrader → FactExtractor → Synthesizer → CompletenessChecker → [loop back to ResearchWorker ≤3×] → HallucinationGrader) → Answer

### Notes
- `constants.py` defines collection names (`COLLECTION_NAME_DROUGH = 'drough'`)
- Tests in `tests/rag/` require infrastructure running and a populated Qdrant collection
- `DroughtEtl` is retained for the climate dataset but `GeneralEtl` is the active default
- Reranker top-N is adaptive via `ModelParams.create_from_context_window()` — not a fixed 10
- Re-indexing required when changing chunk parameters or switching embedding models: `docker compose run --rm app --run-etl --erase --path /data/input/`
- DuckDB file is stored at `data/sql/tabular.duckdb` (in Docker: persisted in `duckdb_data` named volume); `--erase` drops all DuckDB tables alongside the Qdrant collection
- ETL input files: place in `./data/input/` on the host (bind-mounted read-only to `/data/input` in the container); for ad-hoc paths: `docker compose run --rm -v /your/path:/data/input:ro app --run-etl --path /data/input/`

### Document Metadata Schema

**Fields on chunk points:**
- `source: str` — filename stem (e.g. `history_of_metal`)
- `file_type: str` — extension without dot (e.g. `pdf`, `csv`)
- `header_path: str` — H1 > H2 > H3 path of the chunk
- `is_table: bool` — True for table-extracted chunks
- `chunk_index: int` — position within the source file

**Additional fields on CSV/XLSX row points:**
- `row_index: int` — original row number
- + one key per column with its typed value (int/float/str)

### RAG Agent — Node Details

**Router** — Classifies query intent: `general`, `rag`, `rag_exhaustive`, or `rag_summarization`. Uses keyword fallback (`_keyword_intent_upgrade`) for small models that miss exhaustive/summarization patterns.

**QueryPlanner** — Selects a retrieval strategy (`VECTOR`, `SQL`, `HYBRID`, `SCROLL`) and generates vector search queries. Detects aggregation intent and identifies the target source/dataset. Routes to the appropriate downstream node.

**AnalyticalQuery** — Generates and executes a SQL query against DuckDB using `get_compact_catalog()` for schema context. For `HYBRID` strategy, passes SQL results to ResearchWorker for further vector refinement.

**ScrollRetriever** — Fetches all chunks from a specific source (used for `SCROLL` strategy / summarization queries), then reranks the top-N and passes to FactExtractor.

**ResearchWorker** — Parallel hybrid search workers in Qdrant, one per query. Strategy waterfall: Strict (year+location) → Year-only → Location-only → Pure Vector Search. Fetches up to 50 docs per strategy, stops when sufficient.

**RetrievalGrader (reranker)** — Uses FlashRank (`ms-marco-MiniLM-L-12-v2`) to rerank all accumulated `search_results` by relevance to the original question. Deduplicates by doc ID, keeps top 10. Top-N is adaptive based on model context window via `ModelParams`.

**FactExtractor** — Batched LLM extraction: for each batch of top docs, asks the LLM to extract only verbatim sentences relevant to the query. Includes a `difflib.SequenceMatcher` hallucination guard: if the compressed output has similarity < 0.15 to the original, the original is kept unchanged.

**Synthesizer** — Builds numbered source list from deduplicated `distilled_facts`, synthesizes with strict citation, faithfulness, and temporal accuracy rules.

**CompletenessChecker** — After synthesis, checks if the accumulated context is sufficient to fully answer the question. If not (and `retrieval_iterations < 3`), generates a keyword-focused follow-up query and loops back to ResearchWorker. Key behaviours:
- `distilled_facts` accumulates across all gap iterations — synthesizer sees everything found
- Deduplicates by doc ID before evaluating context
- Injects previous follow-up query into prompt so model generates a different angle
- Detects if model repeats the same query → immediately stops the loop
- `follow_up_query` must be keywords/proper nouns only — no question words, no references to "the context"

**HallucinationGrader** — Verifies answer is grounded in the distilled facts. Up to 2 retries before returning a fallback error message.

### RAG Prompt Design
Key rules currently in effect in `rag/agents/prompts.py`:
- **COMPLETENESS RULE** (Synthesizer): For enumeration/listing questions, compile from *all* retrieved sources rather than stopping at the first match
- **TEMPORAL ACCURACY RULE** (Synthesizer): Only include facts explicitly associated with the requested year
- **GROUNDING RULE** (Synthesizer): Refuse only when *none* of the sources are topically relevant — do not refuse because the answer is partial
- **VERBATIM RULE** (FactExtractor): When uncertain, return the document unchanged — never paraphrase or summarise
- **VECTOR QUERY RULE** (CompletenessChecker): `follow_up_query` must be keyword-based for vector search — no question words, no self-referential phrases

### Test Report Metrics
`test_rag_custom.py` computes ML metrics per test session using a 2D confusion matrix where TP = relevancy ≥ 4 AND faithfulness ≥ 4:
- `accuracy`, `precision`, `recall`, `f1` — stored under `metadata.ml_metrics` in the judgement report JSON
- `success_rate` — percentage of questions scoring ≥ 4 on both axes

---

## Possible Future Improvements

### 1. Fact Extractor — Remove or Replace
The LLM-based extractor adds significant latency and fails frequently with smaller models that generate summaries instead of extracting verbatim text. The `difflib` similarity guard already catches most hallucinations, but that means extraction is bypassed for a large share of docs anyway.

**Options:**
- **Remove entirely** — pass full reranked chunks directly to synthesizer; simplest and most reliable
- **Rule-based sentence extractor** — select sentences containing query keywords using BM25 or TF-IDF; zero hallucination risk, near-zero latency
- **Raise minimum doc length** — only extract if `len(doc.text) > 800`; skip short, already-concise chunks

### 2. Smarter Reranking Cutoff
The reranker currently uses an adaptive top-N based on context window size, but still applies a hard cap regardless of score distribution. Improvements:
- Score-gap detection: keep all docs within X% of the top score rather than a fixed count
- Adaptive cutoff: for listing/enumeration questions, raise cap further

### 3. Metadata-Aware Numeric Filtering
For CSV/XLSX sources, the ETL already stores typed numeric values as Qdrant payload fields. The QueryPlanner could identify when a query targets a numeric condition (`review > 9`) and build a Qdrant `Range` filter directly, bypassing similarity entirely for the numeric dimension.

### 4. Hallucination Grader Improvement
The current grader uses a binary yes/no LLM call which is unreliable for small models. Alternatives:
- **NLI-based grader**: use a local Natural Language Inference model (e.g. `cross-encoder/nli-deberta-v3-small`) to score each claim in the answer against the source passages — deterministic, fast, no LLM call needed
- **Sentence-level citation check**: for each sentence in the answer, verify that a supporting source sentence exists above a cosine similarity threshold
