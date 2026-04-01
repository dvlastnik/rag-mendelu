# AgenticRAG

Agentic Retrieval-Augmented Generation (RAG) system built with Python. The system combines a multi-strategy ETL pipeline with a LangGraph agentic workflow to deliver accurate, grounded answers from a mixed document knowledge base.

## Key Features

* **Modern Tooling:** Built using **[uv](https://github.com/astral-sh/uv)** for fast, reliable Python dependency management.
* **Vector Database:** Utilizes **[Qdrant](https://qdrant.tech/)** for high-performance hybrid search (dense + sparse embeddings via SPLADE).
* **Analytical Database:** Uses **[DuckDB](https://duckdb.org/)** (embedded, no extra service) for SQL-based aggregation and ranking queries on tabular data.
* **Smart Ingestion:** Uses **[Docling](https://github.com/DS4SD/docling)** to accurately parse PDFs, DOCX, and PPTX files into clean Markdown.
* **ETL Pipeline:**
    * Converts documents to Markdown (PDF, DOCX, PPTX, MD, TXT, CSV, XLSX).
    * Extracts Markdown tables and stores them in both Qdrant and DuckDB.
    * Cleans and segments text by logical header sections.
    * **Semantic Chunking:** Splits text based on sentence similarity rather than fixed character counts.
    * **Hybrid Embeddings:** Generates both dense and sparse vectors for optimal retrieval.
* **Agentic RAG:** A LangGraph pipeline with specialized agents (Router, QueryPlanner, ResearchWorker, RetrievalGrader, FactExtractor, Synthesizer, CompletenessChecker, HallucinationGrader) that routes each query to the most effective retrieval strategy.
* **Interactive TUI:** Running the app with no arguments launches an [InquirerPy](https://inquirerpy.readthedocs.io/) wizard that guides you through mode selection, model choice (from `ollama list`), and collection selection — no flags required.

---

## Prerequisites

**To run via Docker (recommended):**
1. **Docker & Docker Compose** — runs both Qdrant and the app container.
2. **[Ollama](https://ollama.com)** — must be installed and running **locally on the host** (not in Docker). This ensures GPU acceleration works natively on all platforms — NVIDIA, AMD (ROCm), and Apple Silicon (Metal).

**To run locally with uv (no app container):**
1. **[uv](https://github.com/astral-sh/uv)** — Python package manager used by this project.
2. **Docker & Docker Compose** — still needed to run Qdrant.
3. **[Ollama](https://ollama.com)** — same as above.

---

## Quick Start

> **The recommended way to use this app is the interactive TUI.** All CLI flags (`--chat`, `--run-etl`, `--ask`, etc.) are also supported for scripting and automation, but the TUI is the primary interface.

### 1. Start Ollama and infrastructure

Ollama runs on your host machine so it can use your GPU natively (NVIDIA, AMD, Apple Silicon):

```bash
# Start ollama
ollama serve

# Pull the default model:
ollama pull ministral-3:8b
```

Then start Qdrant:

```bash
docker compose up -d
```

Check that everything is ready:

```bash
docker compose ps
ollama list   # confirm model is available
```

### 2. Run the interactive TUI

Launch the app with no arguments to start the wizard:

```bash
docker compose run --rm -it app
```

The TUI wizard will guide you through:
1. **Save logs to a file?** — if yes, all logs are written to `/log/YYYYMMDD.log` (e.g. `/log/20260401.log`); terminal stays clean
2. **What to do** — Chat, Ask a question, Run ETL, or Check databases
3. **LLM model** — picked from `ollama list` (or enter manually)
4. **Collection** — select an existing Qdrant collection or create a new one
5. Mode-specific options (file path, erase flag, embedding model, etc.)

When you choose **Chat** or **Ask**, a rich terminal chat interface launches:
- Header shows the active model name
- `.` `..` `...` animation plays while the RAG pipeline is running
- Response is displayed under `Assistant (model):` with a numbered **Sources** list

When you choose **Run ETL**, place your files in `./data/input/` first — the wizard will prompt for the path inside the container (`/data/input/`).

Supported formats: `.pdf`, `.docx`, `.pptx`, `.md`, `.txt`, `.csv`, `.xlsx`

### 3. Direct CLI usage (scripting / automation)

All operations are also available as CLI flags, bypassing the TUI entirely:

```bash
# Ingest documents
docker compose run --rm app --run-etl --path /data/input/

# Interactive chat
docker compose run --rm -it app --chat --collection-name my_collection

# Single question
docker compose run --rm -it app --ask "What is X?" --collection-name my_collection

# Check database status
docker compose run --rm app --check-dbs
```

---

## Usage

### Ingest data (ETL)

```bash
# Ingest a single file or entire folder
docker compose run --rm app --run-etl --path /data/input/myfile.pdf

# Erase the existing collection first, then ingest
docker compose run --rm app --run-etl --erase --path /data/input/

# Use a custom embedding model
docker compose run --rm app --run-etl --embed-model BAAI/bge-m3 --path /data/input/

# Specify a custom Qdrant collection name
docker compose run --rm app --run-etl --collection-name MyCollection --path /data/input/
```

#### Getting files into the container

The `./data/input/` folder on your host is bind-mounted read-only as `/data/input` inside the app container:

```bash
# Copy files to the input folder
cp myfile.pdf ./data/input/
cp -r myfolder/ ./data/input/

# Or mount an arbitrary host path directly (no copying needed)
docker compose run --rm -v /absolute/path/to/files:/data/input:ro app --run-etl --path /data/input/
```

### Run chat

```bash
docker compose run --rm -it app --chat --collection-name collection_name

# Use a different LLM model
docker compose run --rm -it app --model llama3.2:3b --chat --collection-name collection_name
```

### Ask

```bash
docker compose run --rm -it app --ask "question" --collection-name collection_name

# Use a different LLM model
docker compose run --rm -it app --model llama3.2:3b --ask "question" --collection-name collection_name
```

### Check database status

```bash
docker compose run --rm app --check-dbs
```

### Teardown

```bash
# Stop containers (data volumes are preserved)
docker compose down

# Stop and delete all data volumes
docker compose down -v
```

---

## Local Development (uv)

Use this approach when you want to run the Python app directly on your machine — no app container, no image builds, instant code changes take effect.

### 1. Start Qdrant only

```bash
docker compose up -d qdrant
```

Qdrant is the only service that still runs in Docker. The app and Ollama run on your host.

### 2. Install Python dependencies

```bash
uv sync
```

This creates a `.venv` in the project root and installs all dependencies from `uv.lock`.

### 3. Pull your Ollama model

```bash
ollama pull ministral-3:8b
```

### 4. Launch the TUI

```bash
uv run main.py
```

Works identically to the Docker TUI. File paths entered in the ETL wizard are real host paths — no bind-mounts or copying needed.

### 5. Common commands

```bash
# Interactive TUI (no args)
uv run main.py

# Chat directly
uv run main.py --chat --collection-name my_collection

# Ingest a file or folder (use real host paths)
uv run main.py --run-etl --path /path/to/your/files/

# Single question
uv run main.py --ask "What is X?" --collection-name my_collection

# Use a custom embedding model
uv run main.py --embed-model BAAI/bge-m3 --chat

# Check databases
uv run main.py --check-dbs
```

### 6. Run tests

```bash
# Unit tests — no infrastructure needed
uv run pytest tests/agent/

# RAG quality tests — requires running Qdrant + a populated collection
uv run pytest tests/rag/ --model ministral-3:8b --questions tests/rag/questions/questions.json
```

### Environment variables

The same `.env` variables apply. For local dev, `QDRANT_HOST` defaults to `localhost` and `OLLAMA_HOST` defaults to `http://localhost:11434` — both correct without any overrides.

---

## Configuration

All settings have sensible defaults — no `.env` file is required for a basic setup.

Create a `.env` file in the root directory to override defaults:

```
OLLAMA_MODEL=ministral-3:8b
COLLECTION_NAME=default_name
QDRANT_REST_PORT=6333
QDRANT_GRPC_PORT=6334
VECTOR_DB_DISTANCE=DOT
LOG_LEVEL=INFO
```

---

## Architecture

### Services

| Service | Image | Role |
|---|---|---|
| `qdrant` | `qdrant/qdrant` | Vector database (hybrid dense + sparse search) |
| `app` | built from `Dockerfile` | Python CLI — invoke with `docker compose run` |

> **Ollama** runs on the host (not in Docker). The app container reaches it via `host.docker.internal:11434`.

### Data Volumes

| Volume | Contents |
|---|---|
| `qdrant_data` | Qdrant vector storage |
| `app_data` | ETL converted Markdown output |
| `duckdb_data` | DuckDB analytical database file |

### Supported File Types

| Extension | Qdrant storage | DuckDB storage | Notes |
|---|---|---|---|
| `.pdf`, `.docx`, `.pptx` | Text chunks + table rows | Extracted tables | Docling converts to Markdown |
| `.md`, `.txt` | Text chunks + table rows | Extracted tables | Processed natively |
| `.csv` | One document per row | Full file as table | All columns stored as typed metadata |
| `.xlsx` | One document per row | Full file as table | All columns stored as typed metadata |

### ETL Flow

1. **Convert**: Docling converts PDF/DOCX/PPTX to Markdown. MD/TXT files are read directly. CSV/XLSX are loaded as DataFrames.
2. **Extract Tables**: Markdown tables are pulled out, each row becomes a separate document with column values as typed metadata. Tables are also registered in DuckDB for SQL queries.
3. **Split**: Remaining text is split by H1-H4 headers into logical sections.
4. **Chunk**: Each section is split using semantic chunking (sentence similarity).
5. **Embed**: Dense (fastembed/sentence-transformers) and sparse (SPLADE) vectors are generated.
6. **Store**: Documents with vectors and metadata are pushed to Qdrant. CSV/XLSX files are also registered in DuckDB.

### Agentic RAG Flow

When you ask a question, a pipeline of agents collaborates:

- **Router**: Classifies the query intent — `general`, `rag`, `rag_exhaustive`, or `rag_summarization`.
- **General Agent**: Answers directly from LLM knowledge for non-retrieval queries.
- **Query Planner**: Selects a retrieval strategy (`VECTOR`, `SQL`, `HYBRID`, `SCROLL`) and generates search queries.
- **Analytical Query Agent**: Executes LLM-generated SQL against DuckDB for aggregation/ranking queries.
- **Research Worker**: Parallel hybrid search workers in Qdrant, one per query.
- **Scroll Retriever**: Fetches all chunks from a specific source (used for summarization and exhaustive queries).
- **Retrieval Grader**: Reranks all retrieved documents with FlashRank and keeps the top results.
- **Fact Extractor**: Extracts only the sentences relevant to the query from each document.
- **Synthesizer**: Generates the final answer with strict source citations.
- **Completeness Checker**: Evaluates whether the answer fully addresses the question. If not, generates a follow-up query and loops back to the Research Worker (up to 3 iterations).
- **Hallucination Grader**: Verifies the answer is grounded in the retrieved context. Retries with the Synthesizer up to 2 times if hallucinations are detected.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        router_agent(router_agent)
        general_agent(general_agent)
        query_planner_agent(query_planner_agent)
        analytical_query_agent(analytical_query_agent)
        research_worker(research_worker)
        retrieval_grader_agent(retrieval_grader_agent)
        fact_extractor_agent(fact_extractor_agent)
        synthesizer_agent(synthesizer_agent)
        completeness_checker_agent(completeness_checker_agent)
        hallucination_grader_agent(hallucination_grader_agent)
        scroll_retriever(scroll_retriever)
        error_agent(error_agent)
        __end__([<p>__end__</p>]):::last

        __start__ --> router_agent;
        analytical_query_agent -. " NodeName.RESEARCH_WORKER " .-> research_worker;
        analytical_query_agent -. " NodeName.SYNTHESIZER " .-> synthesizer_agent;
        completeness_checker_agent -. " NodeName.HALLUCINATION_GRADER_AGENT " .-> hallucination_grader_agent;
        completeness_checker_agent -. " NodeName.RESEARCH_WORKER " .-> research_worker;
        fact_extractor_agent --> synthesizer_agent;
        hallucination_grader_agent -.-> __end__;
        hallucination_grader_agent -. " NodeName.SYNTHESIZER " .-> synthesizer_agent;
        query_planner_agent -. " NodeName.ANALYTICAL_QUERY " .-> analytical_query_agent;
        query_planner_agent -. " NodeName.ERROR " .-> error_agent;
        query_planner_agent -. " NodeName.RESEARCH_WORKER " .-> research_worker;
        query_planner_agent -. " NodeName.SCROLL_RETRIEVER " .-> scroll_retriever;
        research_worker --> retrieval_grader_agent;
        retrieval_grader_agent --> fact_extractor_agent;
        router_agent -. " NodeName.GENERAL " .-> general_agent;
        router_agent -. " NodeName.QUERY_PLANNER " .-> query_planner_agent;
        scroll_retriever --> fact_extractor_agent;
        synthesizer_agent --> completeness_checker_agent;
        error_agent --> __end__;
        general_agent --> __end__;

        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
