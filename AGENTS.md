# Repository Guidelines

## Project Overview

This repository is a local document question-answering workbench built with FastAPI and a framework-free web frontend. It supports PDF upload/OCR/indexing, Excel policy-library ingestion, retrieval-augmented chat, streaming model responses, document routing, and source navigation back to the original PDF pages.

Core runtime flow:

1. Users upload a PDF or Excel file through the vanilla frontend in `static/`.
2. `app.py` stores the file under `data/uploads/`, creates a document record, and creates or reuses a conversation.
3. PDF files are rendered to page JPEGs under `data/renders/`, then queued for the background OCR/indexing pipeline in `pdf_qa.py`.
4. Excel files are previewed first; after field configuration, `excel_qa.py` converts rows into policy records and searchable chunks.
5. `storage.py` persists documents, conversations, OCR text, Excel chunks, FTS rows, optional sqlite-vec vector rows, and route profiles in SQLite.
6. Chat requests build a small retrieval context from relevant PDF pages or Excel chunks, call the DashScope-compatible LLM API, stream answer deltas to the browser, then persist messages and source metadata.

Primary capabilities:

- PDF page rendering, asynchronous PaddleOCR indexing, retry handling, and per-page OCR progress.
- SQLite FTS5 plus optional sqlite-vec hybrid retrieval for PDF pages, document profiles, and Excel chunks.
- Multi-document routing using explicit title matches, document profile FTS, and optional vector similarity.
- Excel policy-library ingestion with configurable title/content/filter/source fields and chunking.
- Streaming chat with reasoning deltas, answer deltas, usage summaries, source-page metadata, and PDF.js page navigation.

Important runtime directories and data:

- `data/app.db`: SQLite database for documents, conversations, messages, OCR text, FTS state, vector maps, profiles, and Excel policy chunks.
- `data/uploads/`: uploaded source files. Treat as sensitive runtime data.
- `data/renders/`: rendered PDF page images and LLM-optimized image cache. Treat as generated runtime data.
- `tmp/`: disposable sample PDFs and scratch files for local manual testing.

External services are configured through `.env`: DashScope-compatible chat (`LLM_API_KEY` or `DASHSCOPE_API_KEY`), PaddleOCR async jobs (`PADDLEOCR_TOKEN`), and SiliconFlow embeddings (`SILICONFLOW_API_KEY`).

## Project Structure & Module Organization

This repository is a local document question-answering service built with FastAPI.

- `app.py`: FastAPI application, routes, streaming chat APIs, static mounts, and OCR job orchestration.
- `pdf_qa.py`: PDF rendering, OCR pipeline, retrieval, embeddings, document routing, and LLM request construction.
- `storage.py`: SQLite schema, FTS/vector storage helpers, conversations, documents, and OCR status records.
- `excel_qa.py`: Excel preview, field configuration, policy chunking, hybrid retrieval, and answer request construction.
- `static/`: Vanilla frontend assets. `static/app.js` drives the UI; `static/pdfjs/` contains bundled PDF.js viewer files.
- `data/`: Runtime database, uploads, and render cache. Do not commit generated contents.
- `tmp/`: Local sample PDFs and scratch files. Treat as disposable test data.

## Build, Test, and Development Commands

- `python3 -m venv .venv && source .venv/bin/activate`: create and enter a local virtual environment.
- `pip install -r requirements.txt`: install backend dependencies.
- `uvicorn app:app --reload`: run the development server at `http://127.0.0.1:8000`.
- `uvicorn app:app --host 0.0.0.0 --port 8000`: run a production-like local server.

There is no formal build step; the frontend is served directly from `static/`.

## Coding Style & Naming Conventions

Use Python 3.10+ conventions with 4-space indentation, type hints where practical, and clear snake_case names for functions, variables, and module-level constants. Keep route handlers thin when possible; place retrieval, OCR, and persistence logic in `pdf_qa.py` or `storage.py` instead of expanding `app.py`. Frontend code in `static/app.js` should remain framework-free and use descriptive camelCase names for DOM state and helpers.

## Testing Guidelines

No automated test suite is currently checked in. For changes, add tests under a future `tests/` directory using `pytest` or `unittest`, with names like `test_storage.py` and `test_pdf_qa.py`. Until then, verify manually by starting `uvicorn app:app --reload`, uploading a small PDF from `tmp/`, waiting for indexing, asking a question, and confirming PDF.js page navigation works.

## Commit & Pull Request Guidelines

Git history currently contains only `Initial publish`, so use short imperative commit messages such as `Add OCR retry logging` or `Fix document routing fallback`. Pull requests should include a summary, manual verification steps, linked issues if any, and screenshots or screen recordings for UI changes.

## Security & Configuration Tips

Keep API credentials in `.env`; never commit real `DASHSCOPE_API_KEY`, `PADDLEOCR_TOKEN`, or `SILICONFLOW_API_KEY` values. Runtime files in `data/` may contain uploaded PDFs and extracted text, so treat them as sensitive and keep them out of version control.
