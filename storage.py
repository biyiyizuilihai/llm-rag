from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RENDERS_DIR = DATA_DIR / "renders"
DB_PATH = DATA_DIR / "app.db"
_SQLITE_VEC_LOADED = False
_SQLITE_VEC_LOAD_ATTEMPTED = False
_SQLITE_VEC_WARNED = False
_SQLITE_VEC_MODULE = None
_FTS_STOPWORDS = {
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)


def _warn_sqlite_vec_once(message: str) -> None:
    global _SQLITE_VEC_WARNED
    if _SQLITE_VEC_WARNED:
        return
    print(message)
    _SQLITE_VEC_WARNED = True


def _load_sqlite_vec_extension(conn: sqlite3.Connection) -> None:
    global _SQLITE_VEC_LOAD_ATTEMPTED, _SQLITE_VEC_LOADED, _SQLITE_VEC_MODULE
    if not _SQLITE_VEC_LOAD_ATTEMPTED:
        _SQLITE_VEC_LOAD_ATTEMPTED = True
        try:
            import sqlite_vec

            _SQLITE_VEC_MODULE = sqlite_vec
            _SQLITE_VEC_LOADED = True
        except Exception as exc:  # noqa: BLE001
            _warn_sqlite_vec_once(
                f"[storage] sqlite-vec import failed, continuing with FTS only: {exc}"
            )
            return

    if not _SQLITE_VEC_LOADED or _SQLITE_VEC_MODULE is None:
        return

    try:
        _SQLITE_VEC_MODULE.load(conn)
    except Exception as exc:  # noqa: BLE001
        _SQLITE_VEC_LOADED = False
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec load failed, continuing with FTS only: {exc}"
        )


def sqlite_vec_available() -> bool:
    return _SQLITE_VEC_LOADED


def get_connection() -> sqlite3.Connection:
    ensure_storage()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.enable_load_extension(True)
    except Exception:
        pass
    _load_sqlite_vec_extension(conn)
    return conn


def _ensure_document_ocr_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(documents)").fetchall()
    }
    if "ocr_status" not in existing_columns:
        conn.execute(
            "ALTER TABLE documents ADD COLUMN ocr_status TEXT NOT NULL DEFAULT 'pending'"
        )
    if "ocr_progress" not in existing_columns:
        conn.execute(
            "ALTER TABLE documents ADD COLUMN ocr_progress INTEGER NOT NULL DEFAULT 0"
        )
    if "ocr_detail" not in existing_columns:
        conn.execute(
            "ALTER TABLE documents ADD COLUMN ocr_detail TEXT NOT NULL DEFAULT ''"
        )


def _ensure_messages_metadata_column(conn: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(messages)").fetchall()
    }
    if "metadata_json" not in existing_columns:
        conn.execute(
            "ALTER TABLE messages ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
        )


def _ensure_conversations_document_nullable(conn: sqlite3.Connection) -> None:
    columns = conn.execute("PRAGMA table_info(conversations)").fetchall()
    if not columns:
        return

    document_column = next((row for row in columns if row["name"] == "document_id"), None)
    if document_column is None or int(document_column["notnull"] or 0) == 0:
        return

    conn.executescript(
        """
        CREATE TABLE conversations__new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
        );

        INSERT INTO conversations__new (id, document_id, title, created_at, updated_at)
        SELECT id, document_id, title, created_at, updated_at
        FROM conversations;

        DROP TABLE conversations;
        ALTER TABLE conversations__new RENAME TO conversations;
        """
    )


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                display_name TEXT NOT NULL,
                file_sha256 TEXT NOT NULL UNIQUE,
                storage_path TEXT NOT NULL,
                render_dir TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                version_index INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS page_ocr (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                ocr_text TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE (document_id, page_number)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS page_ocr_fts
            USING fts5(ocr_text, content=page_ocr, content_rowid=id);

            CREATE TABLE IF NOT EXISTS page_vec_map (
                vec_rowid INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_profiles (
                document_id INTEGER PRIMARY KEY,
                profile_status TEXT NOT NULL DEFAULT 'pending',
                profile_detail TEXT NOT NULL DEFAULT '',
                summary_text TEXT NOT NULL DEFAULT '',
                doc_type TEXT NOT NULL DEFAULT '',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                title_aliases_json TEXT NOT NULL DEFAULT '[]',
                route_text TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS document_profile_fts
            USING fts5(route_text, content=document_profiles, content_rowid=document_id);

            CREATE TABLE IF NOT EXISTS doc_profile_vec_map (
                vec_rowid INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS conversation_documents (
                conversation_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                rank_index INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (conversation_id, document_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            """
        )
        _ensure_conversations_document_nullable(conn)
        _ensure_document_ocr_columns(conn)
        _ensure_messages_metadata_column(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_page_ocr_document_page ON page_ocr(document_id, page_number)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_page_vec_map_document_page ON page_vec_map(document_id, page_number)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_documents_rank ON conversation_documents(conversation_id, rank_index)"
        )
        if sqlite_vec_available():
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS page_ocr_vec
                USING vec0(embedding FLOAT[1024])
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS doc_profile_vec
                USING vec0(embedding FLOAT[1024])
                """
            )


def _document_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = {
        "id": row["id"],
        "file_name": row["file_name"],
        "display_name": row["display_name"],
        "file_sha256": row["file_sha256"],
        "storage_path": row["storage_path"],
        "render_dir": row["render_dir"],
        "page_count": row["page_count"],
        "version_index": row["version_index"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if "ocr_status" in row.keys():
        payload["ocr_status"] = row["ocr_status"]
    if "ocr_progress" in row.keys():
        payload["ocr_progress"] = int(row["ocr_progress"] or 0)
    if "ocr_detail" in row.keys():
        payload["ocr_detail"] = row["ocr_detail"] or ""
    if "profile_status" in row.keys():
        payload["profile_status"] = row["profile_status"] or "pending"
    if "profile_detail" in row.keys():
        payload["profile_detail"] = row["profile_detail"] or ""
    if "summary_text" in row.keys():
        payload["summary_text"] = row["summary_text"] or ""
    if "doc_type" in row.keys():
        payload["doc_type"] = row["doc_type"] or ""
    if "keywords_json" in row.keys():
        try:
            payload["keywords"] = json.loads(row["keywords_json"] or "[]")
        except Exception:
            payload["keywords"] = []
    if "title_aliases_json" in row.keys():
        try:
            payload["title_aliases"] = json.loads(row["title_aliases_json"] or "[]")
        except Exception:
            payload["title_aliases"] = []
    return payload


def _conversation_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = {
        "id": row["id"],
        "document_id": row["document_id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if "document_name" in row.keys():
        payload["document_name"] = row["document_name"]
    if "document_display_name" in row.keys():
        payload["document_display_name"] = row["document_display_name"]
    if "document_version_index" in row.keys():
        payload["document_version_index"] = row["document_version_index"]
    return payload


def _message_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
    if "metadata_json" in row.keys():
        try:
            payload["metadata"] = json.loads(row["metadata_json"] or "{}")
        except Exception:
            payload["metadata"] = {}
    return payload


def _prepare_fts_query(question: str) -> str:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_./+-]*", question)
        if len(token) > 2 and token.lower() not in _FTS_STOPWORDS
    ]
    if not tokens:
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", question)
            if token
        ]

    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        deduped.append(token)
        seen.add(token)

    if not deduped:
        return ""

    cjk_tokens = [token.strip() for token in re.findall(r"[\u4e00-\u9fff]{2,}", question) if token.strip()]
    for token in cjk_tokens:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
        if len(token) >= 4:
            for size in (4, 5, 6):
                if len(token) <= size:
                    continue
                for start in range(0, len(token) - size + 1):
                    fragment = token[start : start + size]
                    if fragment in seen:
                        continue
                    deduped.append(fragment)
                    seen.add(fragment)

    return " OR ".join(f'"{token.replace(chr(34), " ")}"' for token in deduped[:16])


def list_documents() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                d.*,
                dp.profile_status,
                dp.profile_detail,
                dp.summary_text,
                dp.doc_type,
                dp.keywords_json,
                dp.title_aliases_json
            FROM documents d
            LEFT JOIN document_profiles dp ON dp.document_id = d.id
            ORDER BY d.updated_at DESC, d.id DESC
            """
        ).fetchall()
    return [_document_to_dict(row) for row in rows]


def get_document(document_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                d.*,
                dp.profile_status,
                dp.profile_detail,
                dp.summary_text,
                dp.doc_type,
                dp.keywords_json,
                dp.title_aliases_json
            FROM documents d
            LEFT JOIN document_profiles dp ON dp.document_id = d.id
            WHERE d.id = ?
            """,
            (document_id,),
        ).fetchone()
    return _document_to_dict(row)


def get_document_by_sha(file_sha256: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                d.*,
                dp.profile_status,
                dp.profile_detail,
                dp.summary_text,
                dp.doc_type,
                dp.keywords_json,
                dp.title_aliases_json
            FROM documents d
            LEFT JOIN document_profiles dp ON dp.document_id = d.id
            WHERE d.file_sha256 = ?
            """,
            (file_sha256,),
        ).fetchone()
    return _document_to_dict(row)


def next_document_version(file_name: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(version_index), 0) AS max_version FROM documents WHERE file_name = ?",
            (file_name,),
        ).fetchone()
    return int(row["max_version"]) + 1


def create_document(
    file_name: str,
    display_name: str,
    file_sha256: str,
    storage_path: str,
    render_dir: str,
    page_count: int,
    version_index: int,
) -> dict[str, Any]:
    now = utc_now()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents (
                file_name,
                display_name,
                file_sha256,
                storage_path,
                render_dir,
                page_count,
                version_index,
                created_at,
                updated_at,
                ocr_status,
                ocr_progress,
                ocr_detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_name,
                display_name,
                file_sha256,
                storage_path,
                render_dir,
                page_count,
                version_index,
                now,
                now,
                "pending",
                0,
                "",
            ),
        )
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    return _document_to_dict(row)


def list_conversations() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.*,
                d.file_name AS document_name,
                d.display_name AS document_display_name,
                d.version_index AS document_version_index
            FROM conversations c
            LEFT JOIN documents d ON d.id = c.document_id
            ORDER BY c.updated_at DESC, c.id DESC
            """
        ).fetchall()
    return [_conversation_to_dict(row) for row in rows]


def get_conversation(conversation_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                c.*,
                d.file_name AS document_name,
                d.display_name AS document_display_name,
                d.version_index AS document_version_index
            FROM conversations c
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return _conversation_to_dict(row)


def latest_conversation_for_document(document_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                c.*,
                d.file_name AS document_name,
                d.display_name AS document_display_name,
                d.version_index AS document_version_index
            FROM conversations c
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE c.document_id = ?
            ORDER BY c.updated_at DESC, c.id DESC
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()
    return _conversation_to_dict(row)


def list_conversation_ids_for_document(document_id: int) -> list[int]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id FROM conversations WHERE document_id = ? ORDER BY id ASC",
            (document_id,),
        ).fetchall()
    return [int(row["id"]) for row in rows]


def create_conversation(document_id: int | None, title: str) -> dict[str, Any]:
    now = utc_now()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO conversations (document_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, title, now, now),
        )
        row = conn.execute(
            """
            SELECT
                c.*,
                d.file_name AS document_name,
                d.display_name AS document_display_name,
                d.version_index AS document_version_index
            FROM conversations c
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (cursor.lastrowid,),
        ).fetchone()
    return _conversation_to_dict(row)


def delete_conversation(conversation_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
    return cursor.rowcount > 0


def save_page_ocr(document_id: int, page_number: int, ocr_text: str) -> None:
    payload = ocr_text.strip()
    with get_connection() as conn:
        existing = conn.execute(
            """
            SELECT id, ocr_text
            FROM page_ocr
            WHERE document_id = ? AND page_number = ?
            """,
            (document_id, page_number),
        ).fetchone()
        if existing is not None:
            conn.execute(
                """
                INSERT INTO page_ocr_fts(page_ocr_fts, rowid, ocr_text)
                VALUES('delete', ?, ?)
                """,
                (int(existing["id"]), str(existing["ocr_text"] or "")),
            )
        conn.execute(
            """
            INSERT INTO page_ocr (document_id, page_number, ocr_text)
            VALUES (?, ?, ?)
            ON CONFLICT(document_id, page_number)
            DO UPDATE SET ocr_text = excluded.ocr_text
            """,
            (document_id, page_number, payload),
        )
        row = conn.execute(
            """
            SELECT id
            FROM page_ocr
            WHERE document_id = ? AND page_number = ?
            """,
            (document_id, page_number),
        ).fetchone()
        if row is None:
            return
        rowid = int(row["id"])
        conn.execute(
            "INSERT INTO page_ocr_fts(rowid, ocr_text) VALUES (?, ?)",
            (rowid, payload),
        )


def save_page_vector(document_id: int, page_number: int, embedding: list[float]) -> None:
    if not sqlite_vec_available():
        return

    try:
        import sqlite_vec
    except Exception as exc:  # noqa: BLE001
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec import failed, continuing with FTS only: {exc}"
        )
        return

    with get_connection() as conn:
        existing = conn.execute(
            """
            SELECT vec_rowid
            FROM page_vec_map
            WHERE document_id = ? AND page_number = ?
            """,
            (document_id, page_number),
        ).fetchone()
        if existing is not None:
            vec_rowid = int(existing["vec_rowid"])
            conn.execute("DELETE FROM page_ocr_vec WHERE rowid = ?", (vec_rowid,))
            conn.execute("DELETE FROM page_vec_map WHERE vec_rowid = ?", (vec_rowid,))

        cursor = conn.execute(
            "INSERT INTO page_ocr_vec(embedding) VALUES (?)",
            (sqlite_vec.serialize_float32(embedding),),
        )
        conn.execute(
            """
            INSERT INTO page_vec_map (vec_rowid, document_id, page_number)
            VALUES (?, ?, ?)
            """,
            (int(cursor.lastrowid), document_id, page_number),
        )


def fts_search_pages(question: str, document_id: int, top_k: int = 4) -> list[int]:
    query = _prepare_fts_query(question)
    if not query:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.page_number
            FROM page_ocr_fts f
            JOIN page_ocr p ON p.id = f.rowid
            WHERE f.ocr_text MATCH ? AND p.document_id = ?
            ORDER BY bm25(page_ocr_fts), p.page_number
            LIMIT ?
            """,
            (query, document_id, top_k),
        ).fetchall()
    return [int(row["page_number"]) for row in rows]


def phrase_search_pages(question: str, document_id: int, top_k: int = 4) -> list[int]:
    query = " ".join(str(question or "").split())
    if not query:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT page_number
            FROM page_ocr
            WHERE document_id = ?
              AND lower(ocr_text) LIKE lower(?)
            ORDER BY page_number
            LIMIT ?
            """,
            (document_id, f"%{query}%", top_k),
        ).fetchall()
    return [int(row["page_number"]) for row in rows]


def get_page_ocr_text_map(
    document_id: int,
    page_numbers: list[int] | None = None,
) -> dict[int, str]:
    with get_connection() as conn:
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
                (document_id, *page_numbers),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT page_number, ocr_text
                FROM page_ocr
                WHERE document_id = ?
                ORDER BY page_number
                """,
                (document_id,),
            ).fetchall()
    return {int(row["page_number"]): str(row["ocr_text"] or "") for row in rows}


def vector_search_pages(embedding: list[float], document_id: int, top_k: int = 4) -> list[int]:
    if not sqlite_vec_available():
        return []

    try:
        import sqlite_vec
    except Exception as exc:  # noqa: BLE001
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec import failed, continuing with FTS only: {exc}"
        )
        return []

    try:
        query_blob = sqlite_vec.serialize_float32(embedding)
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT m.page_number
                FROM page_ocr_vec v
                JOIN page_vec_map m ON m.vec_rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                  AND m.document_id = ?
                ORDER BY distance
                """,
                (query_blob, top_k, document_id),
            ).fetchall()
        return [int(row["page_number"]) for row in rows]
    except Exception as exc:  # noqa: BLE001
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec query failed, continuing with FTS only: {exc}"
        )
        return []


def update_ocr_status(
    document_id: int,
    status: str,
    progress: int | None = None,
    detail: str | None = None,
) -> None:
    assignments = ["ocr_status = ?"]
    params: list[Any] = [status]

    if progress is not None:
        assignments.append("ocr_progress = ?")
        params.append(max(0, min(100, int(progress))))
    if detail is not None:
        assignments.append("ocr_detail = ?")
        params.append(detail.strip())

    with get_connection() as conn:
        conn.execute(
            f"UPDATE documents SET {', '.join(assignments)} WHERE id = ?",
            (*params, document_id),
        )


def fail_inflight_ocr_jobs(detail: str) -> int:
    payload = detail.strip()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE documents
            SET ocr_status = 'failed',
                ocr_progress = 0,
                ocr_detail = ?
            WHERE ocr_status = 'processing'
            """,
            (payload,),
        )
    return int(cursor.rowcount or 0)


def get_ocr_status(document_id: int) -> str:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT ocr_status FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if row is None:
        return "pending"
    return str(row["ocr_status"] or "pending")


def delete_page_ocr(document_id: int) -> None:
    with get_connection() as conn:
        rowids = [
            (int(row["id"]), str(row["ocr_text"] or ""))
            for row in conn.execute(
                "SELECT id, ocr_text FROM page_ocr WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        ]
        vec_rowids = [
            int(row["vec_rowid"])
            for row in conn.execute(
                "SELECT vec_rowid FROM page_vec_map WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        ]

        if rowids:
            conn.executemany(
                """
                INSERT INTO page_ocr_fts(page_ocr_fts, rowid, ocr_text)
                VALUES('delete', ?, ?)
                """,
                rowids,
            )
        if vec_rowids and sqlite_vec_available():
            conn.executemany(
                "DELETE FROM page_ocr_vec WHERE rowid = ?",
                [(rowid,) for rowid in vec_rowids],
            )
        if vec_rowids:
            conn.executemany(
                "DELETE FROM page_vec_map WHERE vec_rowid = ?",
                [(rowid,) for rowid in vec_rowids],
            )
        conn.execute("DELETE FROM page_ocr WHERE document_id = ?", (document_id,))


def delete_document(document_id: int) -> bool:
    delete_page_ocr(document_id)
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    return cursor.rowcount > 0


def update_conversation_title(conversation_id: int, title: str) -> dict[str, Any] | None:
    now = utc_now()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, now, conversation_id),
        )
        row = conn.execute(
            """
            SELECT
                c.*,
                d.file_name AS document_name,
                d.display_name AS document_display_name,
                d.version_index AS document_version_index
            FROM conversations c
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return _conversation_to_dict(row)


def update_conversation_document(
    conversation_id: int,
    document_id: int | None,
) -> dict[str, Any] | None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET document_id = ?, updated_at = ? WHERE id = ?",
            (document_id, utc_now(), conversation_id),
        )
        row = conn.execute(
            """
            SELECT
                c.*,
                d.file_name AS document_name,
                d.display_name AS document_display_name,
                d.version_index AS document_version_index
            FROM conversations c
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return _conversation_to_dict(row)


def touch_conversation(conversation_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (utc_now(), conversation_id),
        )


def list_messages(conversation_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [_message_to_dict(row) for row in rows]


def count_messages(conversation_id: int, role: str | None = None) -> int:
    with get_connection() as conn:
        if role:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE conversation_id = ? AND role = ?",
                (conversation_id, role),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
    return int(row["total"])


def create_message(
    conversation_id: int,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, metadata_json, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        row = conn.execute(
            "SELECT * FROM messages WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    return _message_to_dict(row)


def save_conversation_documents(conversation_id: int, document_ids: list[int]) -> None:
    now = utc_now()
    deduped: list[int] = []
    seen: set[int] = set()
    for document_id in document_ids:
        normalized = int(document_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    with get_connection() as conn:
        conn.execute(
            "DELETE FROM conversation_documents WHERE conversation_id = ?",
            (conversation_id,),
        )
        if deduped:
            conn.executemany(
                """
                INSERT INTO conversation_documents (conversation_id, document_id, rank_index, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (conversation_id, document_id, rank_index, now)
                    for rank_index, document_id in enumerate(deduped, start=1)
                ],
            )


def list_conversation_documents(conversation_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                d.*,
                dp.profile_status,
                dp.profile_detail,
                dp.summary_text,
                dp.doc_type,
                dp.keywords_json,
                dp.title_aliases_json,
                cd.rank_index
            FROM conversation_documents cd
            JOIN documents d ON d.id = cd.document_id
            LEFT JOIN document_profiles dp ON dp.document_id = d.id
            WHERE cd.conversation_id = ?
            ORDER BY cd.rank_index ASC, d.id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [_document_to_dict(row) | {"rank_index": int(row["rank_index"])} for row in rows]


def update_document_profile(
    document_id: int,
    *,
    profile_status: str,
    profile_detail: str = "",
    summary_text: str = "",
    doc_type: str = "",
    keywords: list[str] | None = None,
    title_aliases: list[str] | None = None,
    route_text: str = "",
) -> None:
    keywords_json = json.dumps(keywords or [], ensure_ascii=False)
    title_aliases_json = json.dumps(title_aliases or [], ensure_ascii=False)
    now = utc_now()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT route_text FROM document_profiles WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if existing is not None:
            conn.execute(
                """
                INSERT INTO document_profile_fts(document_profile_fts, rowid, route_text)
                VALUES('delete', ?, ?)
                """,
                (document_id, str(existing["route_text"] or "")),
            )

        conn.execute(
            """
            INSERT INTO document_profiles (
                document_id,
                profile_status,
                profile_detail,
                summary_text,
                doc_type,
                keywords_json,
                title_aliases_json,
                route_text,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                profile_status = excluded.profile_status,
                profile_detail = excluded.profile_detail,
                summary_text = excluded.summary_text,
                doc_type = excluded.doc_type,
                keywords_json = excluded.keywords_json,
                title_aliases_json = excluded.title_aliases_json,
                route_text = excluded.route_text,
                updated_at = excluded.updated_at
            """,
            (
                document_id,
                profile_status,
                profile_detail,
                summary_text,
                doc_type,
                keywords_json,
                title_aliases_json,
                route_text,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO document_profile_fts(rowid, route_text)
            VALUES (?, ?)
            """,
            (document_id, route_text.strip()),
        )


def save_document_profile_vector(document_id: int, embedding: list[float]) -> None:
    if not sqlite_vec_available():
        return

    try:
        import sqlite_vec
    except Exception as exc:  # noqa: BLE001
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec import failed, continuing with FTS only: {exc}"
        )
        return

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT vec_rowid FROM doc_profile_vec_map WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if existing is not None:
            vec_rowid = int(existing["vec_rowid"])
            conn.execute("DELETE FROM doc_profile_vec WHERE rowid = ?", (vec_rowid,))
            conn.execute("DELETE FROM doc_profile_vec_map WHERE vec_rowid = ?", (vec_rowid,))

        cursor = conn.execute(
            "INSERT INTO doc_profile_vec(embedding) VALUES (?)",
            (sqlite_vec.serialize_float32(embedding),),
        )
        conn.execute(
            """
            INSERT INTO doc_profile_vec_map (vec_rowid, document_id)
            VALUES (?, ?)
            """,
            (int(cursor.lastrowid), document_id),
        )


def phrase_search_document_profiles(question: str, top_k: int = 5) -> list[int]:
    query = " ".join(str(question or "").split())
    if not query:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT document_id
            FROM document_profiles
            WHERE profile_status = 'done'
              AND lower(route_text) LIKE lower(?)
            ORDER BY updated_at DESC, document_id DESC
            LIMIT ?
            """,
            (f"%{query}%", top_k),
        ).fetchall()
    return [int(row["document_id"]) for row in rows]


def fts_search_document_profiles(question: str, top_k: int = 5) -> list[int]:
    query = _prepare_fts_query(question)
    if not query:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT dp.document_id
            FROM document_profile_fts f
            JOIN document_profiles dp ON dp.document_id = f.rowid
            WHERE f.route_text MATCH ?
              AND dp.profile_status = 'done'
            ORDER BY bm25(document_profile_fts), dp.document_id
            LIMIT ?
            """,
            (query, top_k),
        ).fetchall()
    return [int(row["document_id"]) for row in rows]


def vector_search_document_profiles(embedding: list[float], top_k: int = 5) -> list[int]:
    if not sqlite_vec_available():
        return []

    try:
        import sqlite_vec
    except Exception as exc:  # noqa: BLE001
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec import failed, continuing with FTS only: {exc}"
        )
        return []

    try:
        query_blob = sqlite_vec.serialize_float32(embedding)
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT m.document_id
                FROM doc_profile_vec v
                JOIN doc_profile_vec_map m ON m.vec_rowid = v.rowid
                JOIN document_profiles dp ON dp.document_id = m.document_id
                WHERE v.embedding MATCH ? AND k = ?
                  AND dp.profile_status = 'done'
                ORDER BY distance
                """,
                (query_blob, top_k),
            ).fetchall()
        return [int(row["document_id"]) for row in rows]
    except Exception as exc:  # noqa: BLE001
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec query failed, continuing with FTS only: {exc}"
        )
        return []


def list_route_ready_documents() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                d.*,
                dp.profile_status,
                dp.profile_detail,
                dp.summary_text,
                dp.doc_type,
                dp.keywords_json,
                dp.title_aliases_json
            FROM documents d
            JOIN document_profiles dp ON dp.document_id = d.id
            WHERE d.ocr_status = 'done'
              AND dp.profile_status = 'done'
            ORDER BY d.updated_at DESC, d.id DESC
            """
        ).fetchall()
    return [_document_to_dict(row) for row in rows]
