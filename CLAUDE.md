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

# Run tests across all loaded Ollama models (size-sorted, smallest first)
./run_tests.sh                                                          # default collection=drough
./run_tests.sh --collection random --questions tests/rag/questions/questions_random.json
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
- `AgenticRAG(database_service, embedding_service, model_name)` → main entry point
- `AgenticRAG.chat(question)` → returns `{response, sources, rewritten_queries, extracted_data, compressor_results}`
- Graph nodes in `rag/agents/nodes/`: `general_nodes.py` (Router, General), `rag_nodes.py` (QueryRewriter, Extractor, ResearchWorker, RetrievalGrader, ContextCompressor, GapChecker, Synthesizer, HallucinationGrader, Error)
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
- `tests/rag/questions/` — Question sets: `questions.json` (drough/climate), `questions_random.json` (30 questions over history_of_metal/lotr_lore/hussite_wars/games_2025)
- `run_tests.sh` — Bash script: discovers all loaded Ollama models via `ollama list`, sorts by size (smallest first), runs the full test suite per model, generates `evaluation_matrix.csv`

### Key Data Flow

**ETL (GeneralEtl):** Any file → `BaseEtl.extract()` (converter registry) → Markdown in `data/general/` → `GeneralEtl.transform()` (extract tables, split by H1–H4 headers, clean, semantic/recursive chunk, embed) → Qdrant

**ETL metadata per chunk (documents):** `source` (filename stem), `file_type` (no leading dot, e.g. `pdf`), `header_path`, `headers`, `is_table`, `chunk_index`

**ETL metadata per row (CSV/XLSX):** `source`, `file_type`, `is_table`, `row_index`, `text`, + one key per column with its typed value (int/float/str)

**`_clean_text()` in `GeneralEtl`** strips markdown syntax before embedding: `**bold**`/`__bold__` → plain text, `*italic*` → plain text, `` `code` `` → plain text, `[text](url)` → `text`, `> blockquote` → text, `---` horizontal rules removed. Underscore-italic (`_text_`) is intentionally left alone to avoid corrupting filenames and identifiers.

**RAG query:** User question → Router → (General path OR RAG path: QueryRewriter → ResearchWorker → RetrievalGrader → ContextCompressor → GapChecker → [loop back to ResearchWorker ≤3×] → Synthesizer → HallucinationGrader) → Answer

### Notes
- `constants.py` defines collection names (`COLLECTION_NAME_DROUGH = 'drough'`)
- `--vector-db` flag exists in main.py but only `qdrant` is actively used
- Tests in `tests/rag/` require infrastructure running and a populated Qdrant collection
- `DroughtEtl` is retained for the climate dataset but `GeneralEtl` is the active default
- Reranker in `rag_nodes.py` (`retrieval_grader_agent`) keeps top **10** docs after reranking
- Re-indexing required when changing chunk parameters: `uv run main.py --run-etl --erase --path <folder>`

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

**QueryRewriter** — Generates 2 alternative queries (keyword + conceptual) from the original question, maintaining conversation history for follow-up questions. Always keeps original as query #1.

**ResearchWorker** — Parallel workers (one per query × one per extracted target). Strategy waterfall: Strict (year+location) → Year-only → Location-only → Pure Vector Search. Fetches up to 50 docs per strategy, stops when sufficient.

**RetrievalGrader (reranker)** — Uses FlashRank (`ms-marco-MiniLM-L-12-v2`) to rerank all accumulated `search_results` by relevance to the original question. Deduplicates by doc ID, keeps top 10.

**ContextCompressor** — For each of the top-10 docs, asks the LLM to extract only verbatim sentences relevant to the query. Includes a `difflib.SequenceMatcher` hallucination guard: if the compressed output has similarity < 0.15 to the original, the original is kept unchanged. This catches cases where the LLM generates a summary instead of extracting.

**GapChecker** — After compression, checks if the accumulated context is sufficient to fully answer the question. If not (and `retrieval_iterations < 3`), generates a keyword-focused follow-up search query and loops back to ResearchWorker. Key behaviours:
- `context_compressor_results` accumulates across all gap iterations (via `operator.add`) — synthesizer sees everything found
- Deduplicates by doc ID before evaluating context
- Injects previous follow-up query into prompt so model generates a different angle
- Detects if model repeats the same query → immediately stops the loop
- `follow_up_query` must be keywords/proper nouns only — no question words, no references to "the context"

**Synthesizer** — Builds numbered source list from deduplicated `context_compressor_results`, synthesizes with strict citation, faithfulness, and temporal accuracy rules.

**HallucinationGrader** — Verifies answer is grounded in the compressed context. Up to 2 retries before returning a fallback error message.

### RAG Prompt Design
Key rules currently in effect in `rag/agents/prompts.py`:
- **COMPLETENESS RULE** (Synthesizer): For enumeration/listing questions, compile from *all* retrieved sources rather than stopping at the first match
- **TEMPORAL ACCURACY RULE** (Synthesizer): Only include facts explicitly associated with the requested year
- **GROUNDING RULE** (Synthesizer): Refuse only when *none* of the sources are topically relevant — do not refuse because the answer is partial
- **VERBATIM RULE** (ContextCompressor): When uncertain, return the document unchanged — never paraphrase or summarise
- **VECTOR QUERY RULE** (GapChecker): `follow_up_query` must be keyword-based for vector search — no question words, no self-referential phrases

### Test Report Metrics
`test_rag_custom.py` computes ML metrics per test session using a 2D confusion matrix where TP = relevancy ≥ 4 AND faithfulness ≥ 4:
- `accuracy`, `precision`, `recall`, `f1` — stored under `metadata.ml_metrics` in the judgement report JSON
- `success_rate` — percentage of questions scoring ≥ 4 on both axes

---

## Possible Future Improvements

### 1. Analytical Search Tool (Highest Priority)
**Problem:** Vector search is a "find similar" engine, not "scan and aggregate". Queries like "which game has the highest score?" require MAX/MIN/COUNT over all rows — something that is structurally impossible with top-k vector retrieval.

**Proposed solution:**
- Add `QdrantDbRepository.scroll_all_by_source(source, limit=500)` using Qdrant's scroll API to fetch ALL documents from a named source, bypassing similarity entirely
- Detect aggregation intent keywords in the research_worker or as a new `AnalyticalSearch` node: `highest, lowest, maximum, minimum, most, least, rank, top N, average, count, how many, all X in`
- When aggregation + source detected → scroll all docs, skip the top-10 reranker cap, pass full result set to synthesizer
- Add a `source` field to `ExtractionScheme` so the extractor can identify which dataset the query targets (e.g. `"2025 list"` → `games_2025`)
- Falls back gracefully to normal vector search for non-aggregation queries

### 2. Agent Tools (LangChain/LangGraph Tool Calls)
Convert the agent from a fixed pipeline to a tool-calling agent that can invoke specialized tools on demand:

| Tool | Purpose |
|---|---|
| `vector_search(query, filters)` | Standard hybrid search — what the agent does today |
| `scroll_all(source, filter_field, filter_value)` | Fetch ALL rows from a source; needed for MAX/MIN/COUNT/LIST-ALL queries |
| `numeric_filter_search(source, field, op, value)` | Qdrant payload filter: e.g. `review >= 9.0` on games_2025 |
| `keyword_search(source, keyword)` | Full-text/payload keyword match against a specific source |
| `get_document_outline(source)` | Return header structure of a source file; useful for "what topics does X cover?" |
| `compute_aggregate(source, field, operation)` | Python-side MAX/MIN/AVG/COUNT over all scrolled rows of a source |
| `cross_source_join(query, sources)` | Search multiple named sources and merge results with source attribution |

### 3. Context Compressor — Remove or Replace
The LLM-based compressor adds significant latency (10 LLM calls per query) and fails frequently with smaller models that generate summaries instead of extracting verbatim text. The `difflib` similarity guard already catches most hallucinations, but that means compression is bypassed for ~50% of docs anyway.

**Options:**
- **Remove entirely** — pass full reranked chunks directly to synthesizer; simplest and most reliable
- **Rule-based sentence extractor** — select sentences containing query keywords using BM25 or TF-IDF; zero hallucination risk, near-zero latency
- **Raise minimum doc length** — only compress if `len(doc.text) > 800`; skip short, already-concise chunks

### 4. Smarter Reranking Cutoff
The reranker currently takes a hard top-10 regardless of score distribution. Improvements:
- Score-gap detection: keep all docs within X% of the top score rather than a fixed count
- Adaptive cutoff: for listing/enumeration questions detected at query rewrite time, raise cap to 20–30

### 5. Query Classification at Router
Extend the router to classify not just `rag` vs `general` but also query type:
- `factual` — single-answer lookup → standard vector search
- `aggregation` — MAX/MIN/COUNT/LIST-ALL → analytical search path
- `comparison` — multi-entity contrast → multi-target parallel workers
- `conversational` → general agent

### 6. Metadata-Aware Numeric Filtering
For CSV/XLSX sources, the ETL already stores typed numeric values as Qdrant payload fields. The extractor could identify when a query targets a numeric condition (`review > 9`, `row_index between 10 and 20`) and build a Qdrant `Range` filter directly, bypassing similarity entirely for the numeric dimension.

### 7. Hallucination Grader Improvement
The current grader uses a binary yes/no LLM call which is unreliable for small models. Alternatives:
- **NLI-based grader**: use a local Natural Language Inference model (e.g. `cross-encoder/nli-deberta-v3-small`) to score each claim in the answer against the source passages — deterministic, fast, no LLM call needed
- **Sentence-level citation check**: for each sentence in the answer, verify that a supporting source sentence exists above a cosine similarity threshold
