# Automotive Repair Authorisation — RAG Service

A production-ready **Retrieval-Augmented Generation (RAG)** service that decides whether a vehicle repair job should be authorised, by grounding every decision in **real historical repair records** retrieved from Google BigQuery.

| Component | Technology |
|-----------|------------|
| LLM | Qwen 2.5 via Ollama (local) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector search | FAISS (in-memory, per request) |
| Knowledge source | Google BigQuery (`repair_data` table) |
| Pipeline orchestration | LangGraph |
| API framework | FastAPI + Uvicorn |
| Settings | Pydantic Settings |
| Testing | pytest |

---

## How It Works

The system follows strict RAG principles to prevent hallucination:

1. **Retrieve** — fetch historical repair records for the same vehicle (make + model) from BigQuery, including each job's `repair_description`, `jobLineStatus`, and `job_status_date`.
2. **Augment** — use FAISS to find the past records whose `repair_description` is semantically closest to the incoming `job_title`.
3. **Generate** — send only the retrieved records to the LLM with explicit instructions to base the verdict *solely* on that evidence and cite the specific past job that supports the decision.

The LLM is never allowed to use general automotive knowledge — if no matching authorised history exists, the system returns **NOT AUTHORISED**.

---

## Flow Diagram

### End-to-End Request Flow

```mermaid
flowchart TD
    A[POST /authorised-knowledge\nmake · model · job_title · job_cost] --> B[Build question string\nExtract job_title for retrieval]

    B --> C[RAG Graph — LangGraph]

    subgraph C[RAG Graph — LangGraph]
        direction TB
        L[LOAD\nBigQuery SELECT\nrepair_description · jobLineStatus · job_status_date\nWHERE car_make = ? AND car_model = ?]
        R[RETRIEVE\nFAISS similarity search\nquery = job_title\nreturn top-k chunks]
        G[GENERATE\nLLM prompt with retrieved records\nBase decision SOLELY on records\nCite authorised precedent + date]
        L --> R --> G
    end

    G --> H{Parse JSON response}
    H -->|verdict = AUTHORISED| I[canbeAuthorised = true\nknowledge = reasoning string\ncites past job + date]
    H -->|verdict = NOT AUTHORISED| J[canbeAuthorised = false\nknowledge = reasoning string\nno prior authorisation found]
    H -->|Invalid JSON fallback| K[Scan for VERDICT: marker\ncanbeAuthorised = false if absent]

    I --> Z[AuthorisedKnowledgeResponse]
    J --> Z
    K --> Z
```

### LangGraph Pipeline Detail

```mermaid
flowchart LR
    subgraph BigQuery
        BQ[(biddirect-2\n.bqDataLake\n.repair_data)]
    end

    subgraph LOAD node
        Q1["SELECT car_make, car_model,\ncar_variant, repair_description,\njobLineStatus, job_status_date\nWHERE make=? AND model=?"]
    end

    subgraph RETRIEVE node
        S[Split documents\nRecursiveCharacterTextSplitter]
        F[Build FAISS index\nall-MiniLM-L6-v2 embeddings]
        V["similarity_search(job_title, k=5)\nNOT the full question string"]
        S --> F --> V
    end

    subgraph GENERATE node
        P["Prompt:\n• SOLELY on retrieved records\n• Cite authorised job + date\n• No invented data\n• Respond with JSON only:\n  {reasoning, verdict}"]
        LLM[Qwen via Ollama]
        P --> LLM
    end

    BQ --> Q1 --> S
    V -->|top-k Documents\npage_content=repair_description\nmetadata=jobLineStatus + date| P
```

### Anti-Hallucination Design

```mermaid
flowchart TD
    A{Similar AUTHORISED\nrecord in top-k?} -->|Yes| B[Cite repair description\n+ job_status_date\nVERDICT: AUTHORISED]
    A -->|No AUTHORISED record| C[State: no prior authorisation found\nVERDICT: NOT AUTHORISED]
    A -->|No records at all| D[Insufficient history\nVERDICT: NOT AUTHORISED]

    style B fill:#d4edda,stroke:#28a745
    style C fill:#f8d7da,stroke:#dc3545
    style D fill:#f8d7da,stroke:#dc3545
```

---

## Project Structure

```
ai_rag_project/
│
├── api/                        ← HTTP layer only
│   ├── app.py                  ← FastAPI app factory
│   ├── routes.py               ← Endpoints: /health · /chat · /authorised-knowledge
│   └── schemas.py              ← Pydantic request/response models
│
├── rag/                        ← RAG pipeline
│   ├── document_loader.py      ← BigQuery loader + credential builder
│   └── graph.py                ← LangGraph: load → retrieve → generate
│
├── services/                   ← One file per external concern
│   └── llm_service.py          ← LLMService ABC + OllamaLLMService
│
├── config/
│   └── settings.py             ← All config via env vars / .env
│
├── utils/
│   └── logger.py               ← Structured logging
│
├── tests/
│   ├── test_health.py
│   ├── test_chat.py
│   ├── test_authorised_knowledge.py
│   ├── test_rag_graph.py
│   ├── test_document_loader.py
│   └── test_llm_service.py
│
├── main.py                     ← Uvicorn entry point
├── requirements.txt
├── .env.example                ← Environment variable template
└── .env                        ← Local secrets (gitignored)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service liveness check |
| `POST` | `/chat` | Direct LLM chat (no RAG) |
| `POST` | `/authorised-knowledge` | RAG-powered repair authorisation |

### `POST /authorised-knowledge`

Checks historical repair records and decides whether a new repair job should be authorised.

**Request**
```json
{
  "make":      "Ford",
  "model":     "Ford Focus",
  "job_title": "Brake pad replacement",
  "job_cost":  150.00
}
```

**Response**
```json
{
  "knowledge":        "A brake pad replacement was authorised on 2024-03-15 for this vehicle. This job matches that precedent. VERDICT: AUTHORISED",
  "reason":           "A brake pad replacement was authorised on 2024-03-15 for this vehicle. This job matches that precedent.",
  "canbeAuthorised":  true
}
```

- `knowledge` — full LLM response
- `reason` — text before the VERDICT line; cites the past job and date that justified the decision
- `canbeAuthorised` — `true` only when the last `VERDICT:` in the response is `AUTHORISED` (not `NOT AUTHORISED`)

### `POST /chat`

Direct pass-through to the LLM — no BigQuery or RAG.

**Request**
```json
{ "message": "What is a timing belt?" }
```

**Response**
```json
{ "reply": "A timing belt is..." }
```

---

## Setup

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Runtime |
| Ollama | 0.20+ | Local LLM server |
| GCP Service Account | — | BigQuery access |

### 1. Clone and install

```bash
git clone https://github.com/gowthamik14/ai-rag-project.git
cd ai-rag-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in `.env` with your values — at minimum:

```env
# LLM
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen:latest

# BigQuery
GCP_PROJECT_ID=your-gcp-project-id
BIGQUERY_DATASET=your-dataset-name

# Service account (inline — all fields from your JSON key file)
GCP_SA_TYPE=service_account
GCP_SA_PRIVATE_KEY_ID=your-key-id
GCP_SA_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
GCP_SA_CLIENT_EMAIL=your-sa@your-project.iam.gserviceaccount.com
GCP_SA_CLIENT_ID=your-client-id
GCP_SA_AUTH_URI=https://accounts.google.com/o/oauth2/auth
GCP_SA_TOKEN_URI=https://oauth2.googleapis.com/token
GCP_SA_AUTH_PROVIDER_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
GCP_SA_CLIENT_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/your-sa%40your-project.iam.gserviceaccount.com
GCP_UNIVERSE_DOMAIN=googleapis.com
```

> The service account needs the **`BigQuery`** OAuth scope (`https://www.googleapis.com/auth/bigquery`). The `bigquery.readonly` scope is insufficient — BigQuery uses `jobs.insert` even for `SELECT` queries.

### 3. Install and start Ollama

```bash
brew install ollama
ollama pull qwen:latest
```

> **macOS Metal GPU crash fix** — Ollama 0.20.x has a Metal shader bug. Start with both CPU flags:
> ```bash
> pkill -f ollama
> OLLAMA_NO_GPU=1 OLLAMA_LLM_LIBRARY=cpu ollama serve
> ```

### 4. Run the service

```bash
# Terminal 1 — Ollama
OLLAMA_NO_GPU=1 OLLAMA_LLM_LIBRARY=cpu ollama serve

# Terminal 2 — API
python main.py
```

API: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

---

## BigQuery Table Schema

The service reads from `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.repair_data`.

| Column | Type | Usage |
|--------|------|-------|
| `car_make` | STRING | Filter — matches request `make` |
| `car_model` | STRING | Filter — matches request `model` |
| `car_variant` | STRING | Metadata (passed to LLM context) |
| `repair_description` | STRING | Embedded by FAISS for similarity search |
| `jobLineStatus` | STRING | `AUTHORISED` / `NOT AUTHORISED` — drives the verdict |
| `job_status_date` | DATE | Cited in the `reason` when authorising |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `RAG Service` | FastAPI app title |
| `APP_VERSION` | `0.1.0` | App version |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_HOST` | `0.0.0.0` | Bind host |
| `API_PORT` | `8000` | Bind port |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `qwen:latest` | Ollama model tag |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | FAISS top-k results |
| `GCP_PROJECT_ID` | — | GCP project hosting BigQuery |
| `BIGQUERY_DATASET` | — | BigQuery dataset name |
| `GCP_SA_*` | — | Service account fields (see `.env.example`) |

---

## Running Tests

```bash
pytest
```

All tests use fakes and mocks — no live Ollama, BigQuery, or network calls required.

```
tests/test_health.py                 ← GET /health
tests/test_chat.py                   ← POST /chat
tests/test_authorised_knowledge.py   ← POST /authorised-knowledge (HTTP layer)
tests/test_rag_graph.py              ← LangGraph nodes + pipeline
tests/test_document_loader.py        ← BigQuery loader + credential builder
tests/test_llm_service.py            ← OllamaLLMService
```

---

## Running Evaluations

The eval suite runs three offline metrics against a golden dataset of 5 repair cases — no BigQuery connection required.

**Standard run (no Ollama needed):**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m eval.run_eval
```

**With DeepEval faithfulness metric (requires Ollama running):**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m eval.run_eval --deepeval
```

**Save results to a custom file:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m eval.run_eval --output my_results.json
```

Results are written to `eval_results.json` by default.

> **Note:** The `KMP_DUPLICATE_LIB_OK=TRUE` prefix is required on macOS to avoid an OpenMP conflict between PyTorch and FAISS.

### Metrics

| Metric | What it checks | LLM required |
|--------|---------------|--------------|
| **VerdictAccuracy** | `_evaluate_repair()` produces the correct `AUTHORISED` / `NOT AUTHORISED` verdict for each case | No |
| **ReasoningCompleteness** | The reasoning sentence includes the attribution fields (updated-by user and date) from the retrieved records | No |
| **RetrievalRelevance** | Retrieved chunks are keyword-relevant to the job title query (precision ≥ 0.5) | No |
| **DeepEval Faithfulness** | LLM-as-judge checks that the reasoning is grounded in the retrieved context | Yes (Ollama) |

### Golden Dataset

The 5 evaluation cases cover the full decision space:

| Case | Scenario | Expected verdict |
|------|----------|-----------------|
| `authorised_cost_in_range` | Cost within historical range, majority AUTHORISED | AUTHORISED |
| `not_authorised_cost_too_high` | Cost > 2× historical average | NOT AUTHORISED |
| `not_authorised_cost_too_low` | Cost < ½ historical average | NOT AUTHORISED |
| `not_authorised_majority_declined` | Majority of past records DECLINED | NOT AUTHORISED |
| `not_authorised_no_records` | No historical records found | NOT AUTHORISED |

---

## Supported Qwen Models

| Model | Size | Speed (CPU) |
|-------|------|-------------|
| `qwen:latest` (4B Q4) | 2.2 GB | Fast — default |
| `qwen2.5:3b` | 1.9 GB | Fast |
| `qwen2.5:7b` | 4.7 GB | Moderate |
| `qwen2.5:14b` | 9.0 GB | Slow |

Switch model: update `LLM_MODEL` in `.env` and run `ollama pull <model>`.
