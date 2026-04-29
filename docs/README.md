# Project Handoff Notes

This document describes the current state of the local RAG application so a future developer can quickly understand the codebase, runtime behavior, and open issues.

## Current Scope

The project is a local FastAPI-based document Q&A service with a vanilla JavaScript frontend. It now supports two document flows:

- PDF Q&A: upload PDF, OCR/render pages, index extracted text, retrieve relevant pages, answer with page/source references.
- Excel policy Q&A: upload `.xls` / `.xlsx`, preview headers, configure title/content/filter/source fields, index each row as a policy document, chunk policy text, retrieve chunks, and answer with policy sources.

The frontend is served from `static/`. Runtime files live in `data/` and should be treated as sensitive generated data.

## Key Files

- `app.py`: FastAPI routes, upload APIs, conversation APIs, and streaming response handling.
- `pdf_qa.py`: PDF OCR/retrieval/embedding helpers and LLM request construction.
- `excel_qa.py`: Excel parsing, field config normalization, policy row indexing, chunking, query classification, Excel retrieval, and LLM context construction.
- `storage.py`: SQLite schema, FTS tables, sqlite-vec vector tables, documents/conversations/messages persistence.
- `static/app.js`: frontend state management, uploads, Excel config modal, chat streaming, and PDF.js integration.
- `static/app.css`: frontend styling.
- `requirements.txt`: Python dependencies, including `xlrd`, `openpyxl`, and `sqlite-vec`.

## Runtime Commands

Create/install environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app on the currently used local port:

```bash
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Basic health check:

```bash
curl -s http://127.0.0.1:8001/api/health
```

Syntax check after edits:

```bash
.venv/bin/python -m py_compile app.py pdf_qa.py excel_qa.py storage.py
```

## Excel Ingestion Flow

1. User uploads Excel through the existing file upload UI.
2. Backend reads workbook headers and sample rows.
3. Frontend opens a custom Excel configuration modal.
4. User selects:
   - title field
   - content field(s)
   - filter field(s), usually `级别`
   - source/display field(s), usually `政策发文号`
5. Backend stores the config in `documents.config`.
6. Each Excel row becomes one policy in `excel_policies`.
7. Policy text is chunked with fixed overlap by default:
   - chunk size: `1000`
   - overlap: `100`
8. Chunks are stored in `excel_policy_chunks`.
9. SQLite FTS index is populated for keyword search.
10. If embedding is available, chunk vectors are stored in `excel_policy_chunk_vec` plus `excel_chunk_vec_map`.

Current policy Excel test file:

```text
副本政策库汇总表_20260324.xls
```

Known indexed state during development:

```text
713 policy rows
3499 chunks
3499 vector rows
```

## Excel Query Flow

The official Excel answer path should remain:

```text
user question
→ query classifier / rule fallback
→ SQLite FTS top 5
→ vector top 5
→ union by chunk_id
→ add neighboring chunks
→ group by policy
→ send up to 6 candidate policies to LLM
→ answer with sources
```

Important current behavior:

- The current official retrieval strategy is intentionally simple: FTS and vector results are unioned, not weighted.
- Neighbor chunks are marked with `retrieval_sources: ["neighbor"]`.
- Direct hits expose `retrieval_sources` such as `fts` or `vector`.

## Query Classification

There is a small-model classifier hook:

```python
EXCEL_QUERY_CLASSIFIER_MODEL
```

However, in the current local environment this env var is unset, so Excel query planning falls back to `classify_policy_query_by_rule()` in `excel_qa.py`.

The rule fallback currently extracts filters such as:

```text
国家级 / 省级 / 市级 / 区级 / 县级 / 苏州市级 / 江苏省级
```

Known issue: the fallback does **not** normalize aliases against actual metadata values. For example, user text `市级` becomes:

```json
{"级别": "市级"}
```

But the Excel metadata contains:

```text
苏州市级
园区本级
国家级
```

This can cause zero retrieval results.

## PDF Flow

PDF support predates the Excel flow. The pipeline is roughly:

```text
upload PDF
→ render/OCR pages
→ store page text
→ build FTS/vector indexes
→ retrieve pages
→ send page context to LLM
→ answer with page references
```

PDF logic mainly lives in `pdf_qa.py`, while persistence lives in `storage.py`. The frontend uses bundled PDF.js under `static/pdfjs/`.

## Current Known Problems

### 1. Filter Alias Mismatch

User wording can differ from stored metadata values:

- user: `市级`
- data: `苏州市级`

This can produce zero hits. A robust fix should derive allowed metadata values from the indexed Excel document and normalize user aliases before filtering.

Suggested first mapping:

```text
市级 -> 苏州市级
省级 -> 江苏省级
园区 -> 园区本级
国家 -> 国家级
```

Avoid hardcoding too much if future Excel files have different regions.

### 2. Classifier Not Actually Enabled

`EXCEL_QUERY_CLASSIFIER_MODEL` is empty in the current environment, so no small model is being called for query planning. The code path exists, but local behavior is rule-based.

If enabling it, update the system prompt so returned filters use actual metadata values or aliases that the backend can normalize.

### 3. Retrieval Failure Looks Like Message Disappearing

For streaming chat, if backend returns an `error` event, frontend `sendQuestion()` removes the temporary user and assistant messages and puts the question back into the input. This makes failures feel like the question "vanished".

Relevant frontend area:

```text
static/app.js: sendQuestion()
static/app.js: handleStreamEvent()
```

Recommended change: keep the user question in chat and show an assistant error message like "未检索到相关政策片段".

### 4. FTS Noise on Letter Grades

Queries like `E类人才` trigger irrelevant FTS hits because many policy texts contain rating choices like `A/B/C/D/E/F`. Vector retrieval helps, but official results still include noisy candidates because the official path unions FTS top 5 and vector top 5.

The LLM prompt now tells the model to ignore unrelated candidate policies. Further improvements could add alias-aware filters or a lightweight reranker.

### 5. Chunking Is Simple

Excel policies currently use fixed-size overlapping chunks, not article-aware chunking. This is acceptable for MVP but can miss exact context boundaries. Possible future strategy:

```text
第X条 split
→ chapter split
→ paragraph split
→ fixed fallback
```

### 6. Embedding Depends on Network Environment

The running server has successfully generated query embeddings, but direct shell diagnostics may fail if proxy variables point to `127.0.0.1:7890` inside the sandbox. When testing locally, check:

```bash
python -c "import os,requests; print(requests.utils.get_environ_proxies('https://api.siliconflow.cn/v1/embeddings'))"
```

For isolated diagnostics, unset proxy variables if appropriate.

## Database Notes

SQLite database path:

```text
data/app.db
```

Important tables:

- `documents`: uploaded files, config, status, row counts.
- `conversations`, `messages`: chat state.
- `excel_policies`: one row per Excel policy.
- `excel_policy_chunks`: chunk text and metadata.
- `excel_policy_chunks_fts`: SQLite FTS index.
- `excel_policy_chunk_vec`: sqlite-vec vector table.
- `excel_chunk_vec_map`: maps vector row IDs to chunk/policy/document IDs.

Do not commit `data/`; it contains uploaded documents, extracted text, and local runtime state.

## Environment Variables

Common variables:

```text
DASHSCOPE_API_KEY
SILICONFLOW_API_KEY
SILICONFLOW_EMBEDDING_URL
SILICONFLOW_EMBEDDING_MODEL
PADDLEOCR_TOKEN
EXCEL_QUERY_CLASSIFIER_MODEL
```

Keep real credentials in `.env` only. Do not commit them.

## Suggested Development Priorities

1. Add metadata value normalization for Excel filters.
2. Improve frontend error display so failed retrieval does not erase the visible question.
3. Add a diagnostic panel or debug endpoint for query plan, FTS hits, vector hits, final candidate policies, and LLM prompt size.
4. Decide whether the small classifier should be enabled; if yes, constrain outputs to actual metadata enums.
5. Add automated tests around Excel ingestion and query planning.
6. Consider article-aware chunking for policy documents.
7. Keep ES as A/B tooling until its behavior is proven stable across common Chinese policy queries.

## Manual Test Questions

Use these to compare behavior after retrieval changes:

```text
我想知道苏州对于金融机构有没有什么好政策？
新大学生来苏工作有什么补贴？
申报智能工厂该怎么申请
E类人才有没有相关的生活政策
关于市级通用政策的租房补贴，有说要至少交满多少个月的社保吗？
```

For each question, inspect:

- `query_plan.filters`
- `sqlite.policies`
- `sqlite.chunks[].retrieval_sources`
- whether the final answer ignores irrelevant candidate policies
