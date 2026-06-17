# AI-Disaster Chatbot

A multi-agent chatbot focused entirely on **natural disasters** (EM-DAT data) and **clinical text**, composed from:

- **RAG** :: Index disaster-narrative documents derived from the EM-DAT CSVs, with the hybrid Chroma + BM25 retrieval, multimodal-aware ingestion and answer generation.
- **Agent/MCP** :: Orchestration patterns (OpenAI Agents SDK triage + handoffs, FastMCP stdio servers, OpenTelemetry tracing, guardrails) re-used to host **three new MCP servers** for this project.

Added functionality (per the brief): a new Pandas-backed **Disaster MCP server** that answers structured queries about EM-DAT events. The chatbot also ships a **custom Healthcare NL API** behind a ReAct agent.

> Dataset: EM-DAT CSVs in [`dataset/`](dataset/)
> Golden dataset: [`dataset/golden_dataset.json`](dataset/golden_dataset.json).

---

## Table of contents

1. [Focus-area mapping](#focus-area-mapping)
2. [Project layout](#project-layout)
3. [Setup](#setup)
4. [Running the chatbot](#running-the-chatbot)
5. [How the corpus is built](#how-the-corpus-is-built)
6. [Configuration reference](#configuration-reference)
7. [MCP servers and tools](#mcp-servers-and-tools)
8. [Testing](#testing)
9. [Observability](#observability)
10. [Security and guardrails](#security-and-guardrails)
11. [Known limitations](#known-limitations)

---

## Focus-area mapping

The brief calls out four areas to pay extra attention to. Each is realized concretely in code on the disaster domain:

| Focus area | Where it lives |
|---|---|
| **A. Agents, orchestration: Document Processing** | [`DisasterDocumentBuilder`](src/chatbot/disasters/document_builder.py) turns EM-DAT rows into markdown narrative Documents (with impact tables and rich metadata); [`DocumentLoader`](src/chatbot/rag/ingestion/loader.py) chunks them and also picks up any PDFs dropped under `dataset/disaster_reports/`; [`RAGSystem.ingest_disasters`](src/chatbot/rag/rag_system.py) drives the full load -> chunk -> embed -> Chroma + BM25 pipeline. Exposed over MCP as `ingest_corpus` in [`rag_server.py`](src/chatbot/mcp_servers/rag_server.py). |
| **B. ReAct + Custom Healthcare NL tool** | [`HealthcareNLAPI`](src/chatbot/healthcare/nl_api.py) is the custom Healthcare NL API (entity extraction, summarisation, ICD-10 mapping via `with_structured_output`). [`prompts/healthcare_agent.md`](prompts/healthcare_agent.md) enforces an explicit `Thought -> Action -> Observation -> Final Answer` loop on top of the Agents SDK Runner. MCP server: [`healthcare_server.py`](src/chatbot/mcp_servers/healthcare_server.py). |
| **C. Combining Semantic & Keyword Search (Hybrid)** | Dense Chroma + sparse BM25 fused via Reciprocal Rank Fusion (or weighted ranks) in [`hybrid_retriever.py`](src/chatbot/rag/retrieval/hybrid_retriever.py); BM25 with HMAC-SHA256 protected JSON persistence in [`bm25_index.py`](src/chatbot/rag/retrieval/bm25_index.py); fusion configured under `hybrid_search` in [`config/app_config.yaml`](config/app_config.yaml). |
| **D. Multimodal RAG (tables, images)** | Every disaster narrative includes a markdown **impact table** so the RAG corpus exercises table-aware content out of the box. [`multimodal_pdf.py`](src/chatbot/rag/ingestion/multimodal_pdf.py) additionally extracts text + tables (HTML -> markdown) + image regions from any PDF disaster report dropped under `dataset/disaster_reports/`; image bytes are sent to the chat LLM via a base64 `image_url` content block so descriptions are real captions. Backends preferred: `unstructured` -> `pymupdf` -> `pypdf`. |

---

## Project layout

```
ai-final_task/
├── README.md                       # README
├── pyproject.toml                  # deps + pytest config + setuptools src layout
├── .env.example
├── .gitignore
├── config/
│   └── app_config.yaml             # unified config for all layers
├── dataset/
│   ├── 1900_2021_DISASTERS.xlsx - emdat data.csv     # EM-DAT
│   ├── 1970-2021_DISASTERS.xlsx - emdat data.csv     # EM-DAT
│   └── golden_dataset.json         # new disaster-themed eval pairs
├── prompts/
│   ├── triage_agent.md             # routes RAG vs Disaster vs Healthcare
│   ├── rag_agent.md                # narrative search over EM-DAT events
│   ├── disaster_agent.md           # structured Pandas queries over EM-DAT
│   ├── healthcare_agent.md         # explicit ReAct loop instructions
│   ├── rag_answer.md               # grounded-answer prompt for RAG
│   └── evaluate.md                 # disaster-domain RAG eval rubric
├── src/chatbot/
│   ├── main.py                     # main entry point
│   ├── settings/app_config.py      # Pydantic AppConfig + ConfigManager
│   ├── security/                   # guardrails + PII filter
│   ├── model/schemas.py            # Pydantic models (RAG + healthcare)
│   ├── agent/
│   │   ├── orchestrator.py         # AgentSession: triage + 3 MCP servers
│   │   └── tracing.py              # OTel bridge for Agents SDK traces
│   ├── mcp_servers/
│   │   ├── rag_server.py           # NEW MCP wrapper around RAGSystem
│   │   ├── disaster_server.py      # NEW Pandas-backed MCP server
│   │   └── healthcare_server.py    # NEW custom Healthcare NL MCP server
│   ├── rag/                        # vendored from resume_rag (adapted)
│   │   ├── rag_system.py           # ingest disasters + hybrid + answer
│   │   ├── answer_generator.py
│   │   ├── prompts/prompt_manager.py
│   │   ├── ingestion/
│   │   │   ├── loader.py           # chunker + optional PDF report ingestion
│   │   │   ├── multimodal_pdf.py
│   │   │   └── resume_text.py      # generic text utils
│   │   ├── retrieval/{bm25_index,embeddings,hybrid_retriever,vector_store}.py
│   │   ├── security_facade.py      # role-based ACL + audit log
│   │   └── util/json_utils.py
│   ├── disasters/
│   │   ├── repository.py           # Pandas wrapper over EM-DAT CSVs
│   │   └── document_builder.py     # row -> RAG narrative Document
│   └── healthcare/nl_api.py        # custom Healthcare NL API
├── tests/
│   ├── conftest.py                 # shared fixtures (stub LLM, tmp config, ...)
│   ├── fixtures/small_disasters.csv
│   ├── unit/                       # tests
│   └── integration/                # spawns real MCP subprocesses (disaster + healthcare)
└── results/                        # audit, evaluation, log output
```

---

## Setup

### Prerequisites

* Python **3.12+**
* [uv](https://docs.astral.sh/uv/) (recommended) or `pip` + `venv`

### Install

```bash
# from this directory:
uv venv --python 3.12
uv pip install -e ".[dev]"
```


```bash
uv pip install -e ".[dev,multimodal]"
```

### Configure

Copy `.env.example` to `.env` and fill in your credentials:

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Unix
```

`.env` keys:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | API key for chat + embeddings (alias `API_KEY`) |
| `OPENAI_BASE_URL` | Base URL of the proxy (alias `ENDPOINT_URL`) |
| `VECTOR_DB_DIR` | Where Chroma persists (default `./vector-db`) |
| `DATA_DIR` | Dataset directory (default `./dataset`) |
| `RESULTS_DIR` | Audit + logs + eval output (default `./results`) |
| `BM25_HMAC_KEY` | Secret for BM25 index file integrity (HMAC-SHA256) |

---

## Running the chatbot

```bash
.\.venv\Scripts\python -m chatbot.main      # Windows
# or
uv run chatbot                              # console script (any platform)
```

You will see:

```
================================================================
  AI-Disaster Chatbot - RAG | Disasters | Healthcare NL
  Model: gpt-4
  Slash commands: /help /quit /clear /reindex
  (OpenTelemetry tracing enabled - spans printed to console)
================================================================

You:
```

### Slash commands

| Command | Action |
|---|---|
| `/help` | print the command list and example questions |
| `/clear` | reset the in-memory conversation history |
| `/reindex` | ask the RAG agent to call `ingest_corpus(force=true)` |
| `/quit` (also `quit`, `exit`, `q`) | exit the REPL |

### Example interactions

* **Structured disaster query** ::
  *"What are the top 5 countries by total earthquake deaths between 1990
  and 2010?"* -> triage hands off to **Disaster Knowledge Agent** ->
  `disaster_stats(group_by="country", metric="total_deaths",
  disaster_type="Earthquake", year_from=1990, year_to=2010, top_n=5)` ->
  ranked list with Haiti at the top.

* **Narrative disaster query** ::
  *"Tell me about the 2010 Haiti earthquake."* -> triage hands off to
  **Disaster Narrative RAG Agent** -> `answer_with_rag` runs hybrid
  retrieval over indexed EM-DAT narratives and returns a grounded summary
  citing event `2010-0100-HTI`.

* **Healthcare NL** ::
  *"Extract conditions and medications from this note: 'HTN on lisinopril
  10mg daily; HbA1c 7.1%.'"* -> **Healthcare NL Agent** runs an explicit
  ReAct loop calling `extract_medical_entities` and (optionally)
  `link_to_icd10`.

See [`dataset/golden_dataset.json`](dataset/golden_dataset.json) for 12
example queries covering all five categories (disaster_structured,
disaster_narrative, healthcare, edge_case, security).

---

## How the corpus is built

The RAG corpus is built on demand from the EM-DAT repository by
[`DisasterDocumentBuilder`](src/chatbot/disasters/document_builder.py).
Each selected row becomes a single Document with:

* a short narrative header (event id, type, location, dates, magnitude);
* a markdown **impact table** (deaths, injured, affected, damages USD);
* metadata: `id` = EM-DAT `dis_no`, `category` = disaster type,
  `country`, `iso`, `region`, `year`, `headline`.

Selection strategy (configurable via `disasters.indexing_strategy`):

* `recent` (default) :: newest events first.
* `impact` :: highest `total_deaths` first.

The row cap is configurable via `disasters.indexing_max_rows` (default
**500** events) to keep the initial embedding budget reasonable; the full
dataset has ~14 600 rows.

Example narrative (top-impact event):

```
# EM-DAT Event 2010-0100-HTI: Earthquake (Ground movement) in Haiti (2010)

**Type:** Earthquake / Ground movement
**Where:** Port-au-Prince, Haiti (HTI) - Caribbean, Americas
**When:** 2010-01-12

## Impact

| Metric | Value |
| --- | --- |
| Total deaths | 222,570 |
| Injured | 300,572 |
| Affected | 3,700,000 |
| Homeless | 1,500,000 |
| Total affected | 4,000,572 |
| Total damages | $8,000,000,000 USD |
```

---

## Configuration reference

All runtime configuration lives in [`config/app_config.yaml`](config/app_config.yaml).

| Section | Purpose |
|---|---|
| `app` | name, version, log level |
| `llm` | chat model, deployment, temperature, max tokens, timeout |
| `embeddings` | embedding model + batch size |
| `text_splitter` | chunk size + overlap for the document loader |
| `storage` | Chroma persist path, collection name, eval + log subpaths |
| `access_control` | department → disaster-type mapping for RAG ACL |
| `evaluation` | golden dataset path, P@K / R@K, DeepEval thresholds |
| `hybrid_search` | RRF / weighted fusion + BM25 settings |
| `structured_output` | toggle and context budget for grounded answer JSON |
| `document_processing` | text normalisation + multimodal toggles |
| `guardrails` | size limits, blocked-pattern regex, PII redaction, URL schemes |
| `mcp` | path to each MCP server script + spawn timeout |
| `disasters` | which CSVs to load, query caps, **`indexing_max_rows`**, **`indexing_strategy`** |
| `healthcare` | entity types + max input length |
| `agents` | triage / specialist names + prompt filenames (loaded from `prompts/`) |
| `tracing` | OTel exporter list + output redaction caps |
| `logging` | level + format string |

---

## MCP servers and tools

### `disaster_server.py`

| Tool | What it does |
|---|---|
| `query_disasters(...)` | list events with country / type / region / year filters |
| `disaster_stats(group_by, metric, ...)` | aggregates by country / year / region / disaster_type (events, total_deaths, total_affected, total_damages_usd) |
| `top_disasters_by_impact(metric, n=10, ...)` | top N single events by impact |
| `list_disaster_types()` / `list_countries()` | filter discovery |

### `rag_server.py`

| Tool | What it does |
|---|---|
| `hybrid_search(query, k=5, category?)` | RRF-fused dense + BM25 retrieval over disaster narratives |
| `answer_with_rag(query, k=5)` | hybrid retrieval + grounded narrative answer |
| `ingest_corpus(force, max_rows, strategy)` | build narratives from EM-DAT and index them |
| `list_categories()` | disaster categories present in the index |

### `healthcare_server.py`

| Tool | What it does |
|---|---|
| `extract_medical_entities(text)` | structured entities + relations + risk factors + ICD-10 hints |
| `summarize_clinical_text(text, audience)` | summary for `clinician` or `patient` |
| `link_to_icd10(entity)` | up to 3 ICD-10 candidate codes with rationale |

Every tool's output is validated by `validate_tool_output` (size cap +
control-char strip) before being returned to the agent.

---

## Testing

Test coverage is mandatory and the project ships with a comprehensive pytest
suite under [`tests/`](tests/).

```bash
# unit tests only (fast, ~10 s)
uv run pytest tests/unit

# the full suite, including integration tests that spawn real MCP subprocesses
uv run pytest

# with coverage report
uv run pytest --cov=chatbot --cov-report=term-missing
```

* **Unit tests** in `tests/unit/` cover guardrails, PII filter,
  RAG access control (disaster-type categories), text helpers, BM25 index
  (with HMAC-protected persistence), hybrid retriever (RRF + weighted
  fusion), vector-store wrapper (Chroma mocked), answer generator
  (structured LLM stubbed), multimodal PDF extractor (vision LLM
  mocked), disaster repository (Pandas), healthcare NL API (LLM
  stubbed), and JSON utils.
* **Integration tests** in `tests/integration/` spawn the real **disaster**
  and **healthcare** MCP servers as subprocesses and call each tool via
  the MCP client. The healthcare integration test injects a stub LLM in
  the spawned subprocess so no real model call is made. The orchestrator
  integration test mocks `Runner.run` and asserts `AgentSession`
  validates input/output through the guardrails and threads message
  history correctly.

`chatbot.disasters.document_builder` and `chatbot.settings.app_config` are exercised
indirectly (the builder is run as part of `RAGSystem`, and `AppConfig` /
`ConfigManager` are constructed by every fixture); dedicated unit tests for
those two modules and for the Pydantic schemas have been intentionally
omitted.

The MCP server entry-point modules show 0 % in line coverage because pytest
does not trace subprocesses by default; their behaviour is exercised
end-to-end by the integration tests.

Markers:

* `pytest -m "not integration"` runs only the fast unit tests.
* `pytest -m integration` runs only the subprocess tests.

---

## Observability

`install_tracing(cfg.tracing)` in [`tracing.py`](src/chatbot/agent/tracing.py)
bridges the OpenAI Agents SDK tracing API to an OpenTelemetry
`TracerProvider` with a `ConsoleSpanExporter`. Spans for the triage trace,
every agent / tool / generation, and any guardrail violation are exported.
Outputs are PII-redacted and truncated to `tracing.max_output_chars`.
Guardrail errors surface as `guardrail.blocked / guardrail.stage /
guardrail.code` attributes.

Disable tracing via:

```yaml
tracing:
  enabled: false
```

---

## Security and guardrails

[`chatbot.security.guardrails`](src/chatbot/security/guardrails.py) exposes
both an object API (`Guardrails(config)`) and module-level helpers
(`validate_user_input`, `validate_llm_output`, `validate_tool_output`,
`validate_message_history`, `validate_json_write`, `validate_url`,
`sanitize_for_log`). The default instance is installed by `AgentSession`
(and by `chatbot.main`) so the configured `blocked_patterns` and size
limits apply to every code path:

| Layer | Guardrail call |
|---|---|
| REPL prompt | `validate_user_input(query)` |
| Agent run | `validate_user_input`, `validate_llm_output`, `validate_message_history` |
| Each MCP tool return | `validate_tool_output(tool_name, payload)` |
| Hybrid retrieval | `validate_user_input(query)` before any search |
| Answer generation | input validation + post-hoc PII scan + prompt-leak heuristic |
| Audit log writes | sanitised resource string via `sanitize_for_log` |

[`security_facade.py`](src/chatbot/rag/security_facade.py) adds role-based
access control over the indexed corpus (admin / hr_manager / recruiter /
analyst), now mapped to **disaster-type departments** (DisasterResponse,
ClimateOps, PublicHealth, Aerospace). Per-document overrides via
`owner_id` or `access_list` are still supported, and a JSONL audit trail
is written to `results/audit/access_audit.jsonl`.

The BM25 index file on disk is signed with HMAC-SHA256 (key from
`BM25_HMAC_KEY`); a tampered file is refused at load time.

---

## Known limitations

* **No long-term memory.** Conversation history is per-session only.
* **Initial RAG corpus is capped at 500 events** (configurable via
  `disasters.indexing_max_rows`). Indexing all ~14 600 EM-DAT rows is
  possible but consumes substantially more embedding API calls.
* **Vision support assumes the chat model accepts `image_url` content
  blocks.** Older deployments may need a model swap. Errors fall back to a
  placeholder description.
* **Subprocess coverage** of the MCP server scripts is exercised by
  integration tests but not reported by `pytest-cov` (which does not
  follow forks without extra configuration).
* **The OpenAI Agents SDK Runner loop is the implicit ReAct engine.** The
  Healthcare agent prompt forces an *explicit* Thought / Action /
  Observation pattern; the disaster and RAG specialists rely on the SDK's
  tool-calling loop.
