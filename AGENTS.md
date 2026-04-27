# Repository Guidelines

## Project Structure & Module Organization

This repository is a local PDF question-answering service built with FastAPI.

- `app.py`: FastAPI application, routes, streaming chat APIs, static mounts, and OCR job orchestration.
- `pdf_qa.py`: PDF rendering, OCR pipeline, retrieval, embeddings, document routing, and LLM request construction.
- `storage.py`: SQLite schema, FTS/vector storage helpers, conversations, documents, and OCR status records.
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
