from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image


LANGGRAPH_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LLM_RAG_DATA_DIR") or (LANGGRAPH_DIR / "data")).expanduser().resolve()
DB_PATH = DATA_DIR / "app.db"
MAX_TOOL_PAGES = 8
MAX_OCR_TEXT_CHARS = 1800
LLM_RENDER_MAX_PIXELS = 640_000
LLM_RENDER_JPEG_QUALITY = 60
FTS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_document(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "file_name": row["file_name"],
        "display_name": row["display_name"],
        "storage_path": row["storage_path"],
        "render_dir": row["render_dir"],
        "page_count": int(row["page_count"] or 0),
        "file_type": row["file_type"] if "file_type" in row.keys() else "pdf",
        "ocr_status": row["ocr_status"] if "ocr_status" in row.keys() else "pending",
        "ocr_progress": int(row["ocr_progress"] or 0) if "ocr_progress" in row.keys() else 0,
        "ocr_detail": row["ocr_detail"] if "ocr_detail" in row.keys() else "",
    }


def get_document(document_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (int(document_id),)).fetchone()
    return _row_to_document(row)


def list_pdf_documents() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM documents
            WHERE COALESCE(file_type, 'pdf') = 'pdf'
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
    return [doc for row in rows if (doc := _row_to_document(row))]


def _normalize_name(value: str) -> str:
    return re.sub(r"[\s._/()（）\\-]+", "", str(value or "").lower())


def find_pdf_documents_by_name(document_name: str, limit: int = 5) -> list[dict[str, Any]]:
    query = _normalize_name(document_name)
    if not query:
        return []

    matches: list[tuple[dict[str, Any], int]] = []
    for document in list_pdf_documents():
        candidates = [
            str(document.get("display_name") or ""),
            str(document.get("file_name") or ""),
            Path(str(document.get("file_name") or "")).stem,
        ]
        best_score = 0
        for candidate in candidates:
            normalized_candidate = _normalize_name(candidate)
            if not normalized_candidate:
                continue
            if normalized_candidate == query:
                best_score = max(best_score, 1000)
            elif query in normalized_candidate:
                best_score = max(best_score, 800 + min(len(query), 100))
            else:
                query_terms = [term for term in re.split(r"[\s._/()（）\\-]+", document_name.lower()) if len(term) >= 2]
                term_hits = sum(1 for term in query_terms if term in normalized_candidate)
                if term_hits:
                    best_score = max(best_score, term_hits * 100)
        if best_score:
            matches.append((document, best_score))

    matches.sort(key=lambda item: (-item[1], int(item[0]["id"])))
    return [document for document, _ in matches[: max(1, int(limit))]]


def find_pdf_document_payload(document_name: str, limit: int = 5) -> str:
    matches = find_pdf_documents_by_name(document_name, limit=limit)
    if not matches:
        return _json_payload(
            {
                "status": "not_found",
                "document_name": document_name,
                "documents": [],
                "detail": "no PDF document matched the given name",
            }
        )
    return _json_payload(
        {
            "status": "ok",
            "document_name": document_name,
            "document": matches[0],
            "documents": matches,
            "detail": f"found {len(matches)} matching PDF document(s)",
        }
    )


def get_page_ocr_text_map(document_id: int, page_numbers: list[int] | None = None) -> dict[int, str]:
    with _connect() as conn:
        if page_numbers:
            placeholders = ",".join("?" for _ in page_numbers)
            rows = conn.execute(
                f"""
                SELECT page_number, ocr_text
                FROM page_ocr
                WHERE document_id = ?
                  AND page_number IN ({placeholders})
                ORDER BY page_number
                """,
                (int(document_id), *[int(page) for page in page_numbers]),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT page_number, ocr_text
                FROM page_ocr
                WHERE document_id = ?
                ORDER BY page_number
                """,
                (int(document_id),),
            ).fetchall()
    return {int(row["page_number"]): str(row["ocr_text"] or "") for row in rows}


def _clip_text(text: str, limit: int = MAX_OCR_TEXT_CHARS) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _pdf_url(document: dict[str, Any]) -> str:
    storage_path = str(document.get("storage_path") or "").strip()
    if not storage_path:
        return ""
    return f"/pdf-files/{Path(storage_path).name}"


def _tokenize_query(question: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_./+-]*", question):
        normalized = token.lower()
        if len(normalized) <= 2 or normalized in FTS_STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
    for token in re.findall(r"[\u4e00-\u9fff]{2,}", question):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens[:16]


def retrieve_pages(question: str, document_id: int, total_pages: int, max_pages: int) -> list[int]:
    text_map = get_page_ocr_text_map(document_id)
    tokens = _tokenize_query(question)
    scored: list[tuple[int, int]] = []
    for page_number, text in text_map.items():
        normalized_text = text.lower()
        score = 0
        for token in tokens:
            score += normalized_text.count(token.lower())
        if score:
            scored.append((page_number, score))

    if scored:
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [page_number for page_number, _ in scored[:max_pages]]

    fallback_pages = sorted(text_map)[:max_pages]
    if fallback_pages:
        return fallback_pages
    return list(range(1, min(total_pages, max_pages) + 1))


def _resize_image_if_needed(img: Image.Image, max_pixels: int | None) -> Image.Image:
    if not max_pixels:
        return img.convert("RGB")
    width, height = img.size
    pixels = width * height
    if pixels <= max_pixels:
        return img.convert("RGB")
    scale = (max_pixels / pixels) ** 0.5
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)


def _image_to_data_url(image_path: Path) -> str:
    with Image.open(image_path) as img:
        img.load()
        optimized = _resize_image_if_needed(img, LLM_RENDER_MAX_PIXELS)
        from io import BytesIO

        buffer = BytesIO()
        optimized.save(buffer, format="JPEG", quality=LLM_RENDER_JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def load_selected_rendered_page_data_urls(
    render_dir: str | Path,
    page_numbers: list[int],
) -> list[tuple[int, str]]:
    render_path = Path(render_dir)
    result: list[tuple[int, str]] = []
    for page_number in page_numbers:
        image_path = render_path / f"page-{int(page_number):04d}.jpg"
        if not image_path.exists():
            continue
        result.append((int(page_number), _image_to_data_url(image_path)))
    return result


def retrieve_pdf_pages_payload(
    question: str,
    document_id: int | None = None,
    document_name: str | None = None,
    max_pages: int = 4,
) -> str:
    """Return retrieved PDF pages as JSON for agent tool use."""
    normalized_question = " ".join(str(question or "").split())
    if not normalized_question:
        return _json_payload(
            {
                "status": "error",
                "detail": "question is required",
                "hits": [],
            }
        )

    document: dict[str, Any] | None = None
    if document_id is not None:
        document = get_document(int(document_id))
    elif document_name:
        matches = find_pdf_documents_by_name(document_name, limit=5)
        if len(matches) > 1:
            exact_matches = [
                item
                for item in matches
                if _normalize_name(item.get("display_name") or item.get("file_name") or "")
                == _normalize_name(document_name)
            ]
            if len(exact_matches) == 1:
                document = exact_matches[0]
            else:
                return _json_payload(
                    {
                        "status": "ambiguous",
                        "document_name": document_name,
                        "documents": matches,
                        "detail": "multiple PDF documents matched; retry with document_id",
                        "hits": [],
                    }
                )
        elif matches:
            document = matches[0]
    else:
        return _json_payload(
            {
                "status": "error",
                "detail": "document_id or document_name is required",
                "hits": [],
            }
        )

    if not document:
        return _json_payload(
            {
                "status": "not_found",
                "document_id": int(document_id) if document_id is not None else None,
                "document_name": document_name,
                "detail": "document not found",
                "hits": [],
            }
        )

    if str(document.get("file_type") or "pdf") != "pdf":
        return _json_payload(
            {
                "status": "invalid_type",
                "document_id": int(document_id),
                "file_type": document.get("file_type") or "",
                "detail": "document is not a PDF",
                "hits": [],
            }
        )

    ocr_status = str(document.get("ocr_status") or "pending")
    if ocr_status != "done":
        return _json_payload(
            {
                "status": "not_ready",
                "document_id": int(document_id),
                "ocr_status": ocr_status,
                "ocr_progress": int(document.get("ocr_progress") or 0),
                "detail": document.get("ocr_detail") or "PDF index is not ready",
                "hits": [],
            }
        )

    page_limit = max(1, min(int(max_pages or 4), MAX_TOOL_PAGES))
    page_numbers = retrieve_pages(
        question=normalized_question,
        document_id=int(document["id"]),
        total_pages=int(document.get("page_count") or 0),
        max_pages=page_limit,
    )[:page_limit]

    text_map = get_page_ocr_text_map(int(document["id"]), page_numbers=page_numbers)
    rendered_pages = load_selected_rendered_page_data_urls(
        render_dir=document.get("render_dir") or "",
        page_numbers=page_numbers,
    )
    image_map = {int(page_number): data_url for page_number, data_url in rendered_pages}

    hits = [
        {
            "document_id": int(document["id"]),
            "document_name": document.get("file_name") or "",
            "document_display_name": document.get("display_name") or document.get("file_name") or "",
            "page_number": int(page_number),
            "ocr_text": _clip_text(text_map.get(int(page_number), "")),
            "image_data_url": image_map.get(int(page_number), ""),
            "pdf_url": _pdf_url(document),
        }
        for page_number in page_numbers
        if int(page_number) in image_map
    ]

    return _json_payload(
        {
            "status": "ok",
            "document_id": int(document["id"]),
            "question": normalized_question,
            "hits": hits,
            "page_numbers": [hit["page_number"] for hit in hits],
            "detail": (
                f"retrieved {len(hits)} page image(s)"
                if hits
                else "no rendered page image found for retrieved pages"
            ),
        }
    )
