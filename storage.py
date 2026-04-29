from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LLM_RAG_DATA_DIR") or (BASE_DIR / "data")).expanduser().resolve()
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


def normalize_document_runtime_paths(conn: sqlite3.Connection) -> int:
    try:
        rows = conn.execute("SELECT id, storage_path, render_dir FROM documents").fetchall()
    except sqlite3.OperationalError:
        return 0

    changed = 0
    for row in rows:
        updates: dict[str, str] = {}
        storage_path = str(row["storage_path"] or "")
        if storage_path:
            expected_storage_path = UPLOADS_DIR / Path(storage_path).name
            if expected_storage_path.exists() and storage_path != str(expected_storage_path):
                updates["storage_path"] = str(expected_storage_path)

        render_dir = str(row["render_dir"] or "")
        if render_dir:
            expected_render_dir = RENDERS_DIR / Path(render_dir).name
            if expected_render_dir.exists() and render_dir != str(expected_render_dir):
                updates["render_dir"] = str(expected_render_dir)

        if not updates:
            continue

        assignments = ", ".join(f"{column} = ?" for column in updates)
        conn.execute(
            f"UPDATE documents SET {assignments} WHERE id = ?",
            (*updates.values(), int(row["id"])),
        )
        changed += 1

    return changed


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


def _ensure_document_file_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(documents)").fetchall()
    }
    if "file_type" not in existing_columns:
        conn.execute(
            "ALTER TABLE documents ADD COLUMN file_type TEXT NOT NULL DEFAULT 'pdf'"
        )
    if "row_count" not in existing_columns:
        conn.execute(
            "ALTER TABLE documents ADD COLUMN row_count INTEGER NOT NULL DEFAULT 0"
        )
    if "config_json" not in existing_columns:
        conn.execute(
            "ALTER TABLE documents ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'"
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

            CREATE TABLE IF NOT EXISTS excel_policies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_document_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                search_text TEXT NOT NULL,
                source_row INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (file_document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS excel_policy_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id INTEGER NOT NULL,
                file_document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                search_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (policy_id) REFERENCES excel_policies(id) ON DELETE CASCADE,
                FOREIGN KEY (file_document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS excel_policy_chunks_fts
            USING fts5(search_text, content=excel_policy_chunks, content_rowid=id);

            CREATE TABLE IF NOT EXISTS excel_chunk_vec_map (
                vec_rowid INTEGER PRIMARY KEY,
                chunk_id INTEGER NOT NULL UNIQUE,
                policy_id INTEGER NOT NULL,
                file_document_id INTEGER NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES excel_policy_chunks(id) ON DELETE CASCADE,
                FOREIGN KEY (policy_id) REFERENCES excel_policies(id) ON DELETE CASCADE,
                FOREIGN KEY (file_document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            """
        )
        _ensure_conversations_document_nullable(conn)
        _ensure_document_file_columns(conn)
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_excel_policies_file ON excel_policies(file_document_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_excel_chunks_file_policy ON excel_policy_chunks(file_document_id, policy_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_excel_chunk_vec_map_file ON excel_chunk_vec_map(file_document_id, policy_id)"
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
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS excel_policy_chunk_vec
                USING vec0(embedding FLOAT[1024])
                """
            )
        normalize_document_runtime_paths(conn)


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
    if "file_type" in row.keys():
        payload["file_type"] = row["file_type"] or "pdf"
    else:
        payload["file_type"] = "pdf"
    if "row_count" in row.keys():
        payload["row_count"] = int(row["row_count"] or 0)
    else:
        payload["row_count"] = 0
    if "config_json" in row.keys():
        try:
            payload["config"] = json.loads(row["config_json"] or "{}")
        except Exception:
            payload["config"] = {}
    else:
        payload["config"] = {}
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

    for match in re.finditer(r"(?<![A-Za-z])([A-Za-z]{1,2})(?=[\u4e00-\u9fff])", question):
        token = match.group(1).lower()
        if token not in seen:
            deduped.append(token)
            seen.add(token)

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

    if not deduped:
        return ""

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
    file_type: str = "pdf",
    row_count: int = 0,
    config: dict[str, Any] | None = None,
    ocr_status: str = "pending",
    ocr_progress: int = 0,
    ocr_detail: str = "",
) -> dict[str, Any]:
    now = utc_now()
    config_json = json.dumps(config or {}, ensure_ascii=False)
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
                file_type,
                row_count,
                config_json,
                created_at,
                updated_at,
                ocr_status,
                ocr_progress,
                ocr_detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_name,
                display_name,
                file_sha256,
                storage_path,
                render_dir,
                page_count,
                version_index,
                file_type,
                int(row_count or 0),
                config_json,
                now,
                now,
                ocr_status,
                int(ocr_progress or 0),
                ocr_detail,
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


def update_document_ingestion_config(
    document_id: int,
    *,
    config: dict[str, Any],
    row_count: int | None = None,
    status: str | None = None,
    progress: int | None = None,
    detail: str | None = None,
) -> None:
    assignments = ["config_json = ?", "updated_at = ?"]
    params: list[Any] = [json.dumps(config or {}, ensure_ascii=False), utc_now()]

    if row_count is not None:
        assignments.append("row_count = ?")
        params.append(max(0, int(row_count)))
    if status is not None:
        assignments.append("ocr_status = ?")
        params.append(status)
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


def try_claim_ingestion(document_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE documents SET ocr_status = 'processing' WHERE id = ? AND ocr_status != 'processing'",
            (document_id,),
        )
        return cursor.rowcount > 0


def delete_excel_policy_index(document_id: int, conn: sqlite3.Connection | None = None) -> None:
    owns_connection = conn is None
    active_conn = conn or get_connection()
    try:
        vec_rowids = [
            int(row["vec_rowid"])
            for row in active_conn.execute(
                "SELECT vec_rowid FROM excel_chunk_vec_map WHERE file_document_id = ?",
                (document_id,),
            ).fetchall()
        ]
        if vec_rowids and sqlite_vec_available():
            active_conn.executemany(
                "DELETE FROM excel_policy_chunk_vec WHERE rowid = ?",
                [(rowid,) for rowid in vec_rowids],
            )
        if vec_rowids:
            active_conn.executemany(
                "DELETE FROM excel_chunk_vec_map WHERE vec_rowid = ?",
                [(rowid,) for rowid in vec_rowids],
            )

        rows = active_conn.execute(
            """
            SELECT id, search_text
            FROM excel_policy_chunks
            WHERE file_document_id = ?
            """,
            (document_id,),
        ).fetchall()
        if rows:
            active_conn.executemany(
                """
                INSERT INTO excel_policy_chunks_fts(excel_policy_chunks_fts, rowid, search_text)
                VALUES('delete', ?, ?)
                """,
                [(int(row["id"]), str(row["search_text"] or "")) for row in rows],
            )
        active_conn.execute(
            "DELETE FROM excel_policy_chunks WHERE file_document_id = ?",
            (document_id,),
        )
        active_conn.execute(
            "DELETE FROM excel_policies WHERE file_document_id = ?",
            (document_id,),
        )
        if owns_connection:
            active_conn.commit()
    finally:
        if owns_connection:
            active_conn.close()


def save_excel_policy_index(
    document_id: int,
    *,
    config: dict[str, Any],
    policies: list[dict[str, Any]],
) -> int:
    now = utc_now()
    with get_connection() as conn:
        delete_excel_policy_index(document_id, conn=conn)
        for policy in policies:
            metadata = policy.get("metadata") or {}
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            cursor = conn.execute(
                """
                INSERT INTO excel_policies (
                    file_document_id,
                    title,
                    content,
                    metadata_json,
                    search_text,
                    source_row,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    str(policy.get("title") or "").strip(),
                    str(policy.get("content") or "").strip(),
                    metadata_json,
                    str(policy.get("search_text") or "").strip(),
                    int(policy.get("source_row") or 0),
                    now,
                    now,
                ),
            )
            policy_id = int(cursor.lastrowid)
            for chunk in policy.get("chunks") or []:
                chunk_metadata = chunk.get("metadata") or metadata
                chunk_metadata_json = json.dumps(chunk_metadata, ensure_ascii=False)
                chunk_cursor = conn.execute(
                    """
                    INSERT INTO excel_policy_chunks (
                        policy_id,
                        file_document_id,
                        chunk_index,
                        chunk_text,
                        search_text,
                        metadata_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        policy_id,
                        document_id,
                        int(chunk.get("chunk_index") or 0),
                        str(chunk.get("chunk_text") or "").strip(),
                        str(chunk.get("search_text") or "").strip(),
                        chunk_metadata_json,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO excel_policy_chunks_fts(rowid, search_text)
                    VALUES (?, ?)
                    """,
                    (int(chunk_cursor.lastrowid), str(chunk.get("search_text") or "").strip()),
                )

        conn.execute(
            """
            UPDATE documents
            SET config_json = ?,
                row_count = ?,
                ocr_status = 'processing',
                ocr_progress = 70,
                ocr_detail = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(config or {}, ensure_ascii=False),
                len(policies),
                f"Excel 全文索引完成：{len(policies)} 条记录，正在准备向量索引。",
                now,
                document_id,
            ),
        )
    return len(policies)


def list_excel_chunks_for_embedding(document_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.policy_id,
                c.file_document_id,
                c.chunk_index,
                c.chunk_text,
                c.search_text,
                p.title
            FROM excel_policy_chunks c
            JOIN excel_policies p ON p.id = c.policy_id
            WHERE c.file_document_id = ?
            ORDER BY c.id ASC
            """,
            (document_id,),
        ).fetchall()
    return [
        {
            "chunk_id": int(row["chunk_id"]),
            "policy_id": int(row["policy_id"]),
            "document_id": int(row["file_document_id"]),
            "chunk_index": int(row["chunk_index"] or 0),
            "chunk_text": row["chunk_text"] or "",
            "search_text": row["search_text"] or "",
            "title": row["title"] or "",
        }
        for row in rows
    ]


def list_excel_chunks_for_search_index(document_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.policy_id,
                c.file_document_id,
                c.chunk_index,
                c.chunk_text,
                c.search_text,
                c.metadata_json,
                p.title,
                p.source_row,
                d.file_name,
                d.display_name,
                0.0 AS rank
            FROM excel_policy_chunks c
            JOIN excel_policies p ON p.id = c.policy_id
            JOIN documents d ON d.id = c.file_document_id
            WHERE c.file_document_id = ?
            ORDER BY c.id ASC
            """,
            (document_id,),
        ).fetchall()
    return [_excel_chunk_row_to_dict(row) for row in rows]


def save_excel_chunk_vector(
    chunk_id: int,
    policy_id: int,
    document_id: int,
    embedding: list[float],
) -> None:
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
            "SELECT vec_rowid FROM excel_chunk_vec_map WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if existing is not None:
            vec_rowid = int(existing["vec_rowid"])
            conn.execute("DELETE FROM excel_policy_chunk_vec WHERE rowid = ?", (vec_rowid,))
            conn.execute("DELETE FROM excel_chunk_vec_map WHERE vec_rowid = ?", (vec_rowid,))

        cursor = conn.execute(
            "INSERT INTO excel_policy_chunk_vec(embedding) VALUES (?)",
            (sqlite_vec.serialize_float32(embedding),),
        )
        conn.execute(
            """
            INSERT INTO excel_chunk_vec_map (vec_rowid, chunk_id, policy_id, file_document_id)
            VALUES (?, ?, ?, ?)
            """,
            (int(cursor.lastrowid), int(chunk_id), int(policy_id), int(document_id)),
        )


def _excel_chunk_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    try:
        metadata = json.loads(row["metadata_json"] or "{}")
    except Exception:
        metadata = {}
    return {
        "chunk_id": int(row["chunk_id"]),
        "policy_id": int(row["policy_id"]),
        "document_id": int(row["file_document_id"]),
        "document_file_name": row["file_name"],
        "document_display_name": row["display_name"],
        "title": row["title"],
        "source_row": int(row["source_row"] or 0),
        "chunk_index": int(row["chunk_index"] or 0),
        "chunk_text": row["chunk_text"] or "",
        "search_text": row["search_text"] or "",
        "metadata": metadata,
        "rank": float(row["rank"] or 0),
    }


def get_excel_policy_chunks_by_positions(
    positions: list[tuple[int, int]],
    *,
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    normalized_positions = sorted(
        {
            (int(policy_id), int(chunk_index))
            for policy_id, chunk_index in positions
            if int(chunk_index) >= 0
        }
    )
    if not normalized_positions:
        return []

    params: list[Any] = []
    predicates: list[str] = []
    for policy_id, chunk_index in normalized_positions:
        predicates.append("(c.policy_id = ? AND c.chunk_index = ?)")
        params.extend([policy_id, chunk_index])

    filter_sql = ""
    if document_id is not None:
        filter_sql = " AND c.file_document_id = ?"
        params.append(int(document_id))

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.policy_id,
                c.file_document_id,
                c.chunk_index,
                c.chunk_text,
                c.search_text,
                c.metadata_json,
                p.title,
                p.source_row,
                d.file_name,
                d.display_name,
                0.0 AS rank
            FROM excel_policy_chunks c
            JOIN excel_policies p ON p.id = c.policy_id
            JOIN documents d ON d.id = c.file_document_id
            WHERE ({' OR '.join(predicates)})
              AND d.ocr_status = 'done'
              {filter_sql}
            ORDER BY c.file_document_id, c.policy_id, c.chunk_index
            """,
            params,
        ).fetchall()

    return [_excel_chunk_row_to_dict(row) for row in rows]


def search_excel_policy_chunks(
    question: str,
    *,
    document_id: int | None = None,
    filters: dict[str, str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    query = _prepare_fts_query(question)
    filter_items = {
        str(key): str(value).strip()
        for key, value in (filters or {}).items()
        if str(value or "").strip()
    }
    params: list[Any] = []
    filter_sql = ""
    if document_id is not None:
        filter_sql += " AND c.file_document_id = ?"
        params.append(int(document_id))
    for key, value in filter_items.items():
        filter_sql += " AND json_extract(c.metadata_json, ?) = ?"
        params.extend([f"$.{key}", value])

    rows: list[sqlite3.Row] = []
    with get_connection() as conn:
        if query:
            rows = conn.execute(
                f"""
                SELECT
                    c.id AS chunk_id,
                    c.policy_id,
                    c.file_document_id,
                    c.chunk_index,
                    c.chunk_text,
                    c.search_text,
                    c.metadata_json,
                    p.title,
                    p.source_row,
                    d.file_name,
                    d.display_name,
                    bm25(excel_policy_chunks_fts) AS rank
                FROM excel_policy_chunks_fts f
                JOIN excel_policy_chunks c ON c.id = f.rowid
                JOIN excel_policies p ON p.id = c.policy_id
                JOIN documents d ON d.id = c.file_document_id
                WHERE f.search_text MATCH ?
                  AND d.ocr_status = 'done'
                  {filter_sql}
                ORDER BY bm25(excel_policy_chunks_fts), c.file_document_id, c.policy_id, c.chunk_index
                LIMIT ?
                """,
                (query, *params, int(top_k)),
            ).fetchall()

        if not rows:
            like_query = " ".join(str(question or "").split())
            if not like_query:
                return []
            like_terms = [
                term
                for term in re.split(r"[\s,，。；;？?]+", like_query)
                if term
            ][:8]
            like_sql = " AND ".join("lower(c.search_text) LIKE lower(?)" for _ in like_terms)
            like_params = [f"%{term}%" for term in like_terms]
            rows = conn.execute(
                f"""
                SELECT
                    c.id AS chunk_id,
                    c.policy_id,
                    c.file_document_id,
                    c.chunk_index,
                    c.chunk_text,
                    c.search_text,
                    c.metadata_json,
                    p.title,
                    p.source_row,
                    d.file_name,
                    d.display_name,
                    0.0 AS rank
                FROM excel_policy_chunks c
                JOIN excel_policies p ON p.id = c.policy_id
                JOIN documents d ON d.id = c.file_document_id
                WHERE d.ocr_status = 'done'
                  AND {like_sql}
                  {filter_sql}
                ORDER BY c.file_document_id, c.policy_id, c.chunk_index
                LIMIT ?
                """,
                (*like_params, *params, int(top_k)),
            ).fetchall()

    return [_excel_chunk_row_to_dict(row) for row in rows]


def vector_search_excel_policy_chunks(
    embedding: list[float],
    *,
    document_id: int | None = None,
    filters: dict[str, str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if not sqlite_vec_available():
        return []

    try:
        import sqlite_vec
    except Exception as exc:  # noqa: BLE001
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec import failed, continuing with FTS only: {exc}"
        )
        return []

    filter_items = {
        str(key): str(value).strip()
        for key, value in (filters or {}).items()
        if str(value or "").strip()
    }
    params: list[Any] = []
    filter_sql = ""
    if document_id is not None:
        filter_sql += " AND c.file_document_id = ?"
        params.append(int(document_id))
    for key, value in filter_items.items():
        filter_sql += " AND json_extract(c.metadata_json, ?) = ?"
        params.extend([f"$.{key}", value])

    try:
        query_blob = sqlite_vec.serialize_float32(embedding)
        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.id AS chunk_id,
                    c.policy_id,
                    c.file_document_id,
                    c.chunk_index,
                    c.chunk_text,
                    c.search_text,
                    c.metadata_json,
                    p.title,
                    p.source_row,
                    d.file_name,
                    d.display_name,
                    distance AS rank
                FROM excel_policy_chunk_vec v
                JOIN excel_chunk_vec_map m ON m.vec_rowid = v.rowid
                JOIN excel_policy_chunks c ON c.id = m.chunk_id
                JOIN excel_policies p ON p.id = c.policy_id
                JOIN documents d ON d.id = c.file_document_id
                WHERE v.embedding MATCH ? AND k = ?
                  AND d.ocr_status = 'done'
                  {filter_sql}
                ORDER BY distance
                """,
                (query_blob, int(top_k), *params),
            ).fetchall()
        return [_excel_chunk_row_to_dict(row) for row in rows]
    except Exception as exc:  # noqa: BLE001
        _warn_sqlite_vec_once(
            f"[storage] sqlite-vec query failed, continuing with FTS only: {exc}"
        )
        return []


def list_ready_excel_documents() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM documents
            WHERE file_type = 'excel'
              AND ocr_status = 'done'
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
    return [_document_to_dict(row) for row in rows]


def get_excel_filter_enums(document_id: int) -> dict[str, list[str]]:
    doc = get_document(document_id)
    filter_fields = (doc.get("config") or {}).get("filter_fields") or [] if doc else []
    if not filter_fields:
        return {}
    result: dict[str, list[str]] = {}
    with get_connection() as conn:
        for field in filter_fields:
            rows = conn.execute(
                """
                SELECT DISTINCT json_extract(metadata_json, ?) AS val
                FROM excel_policies
                WHERE file_document_id = ?
                  AND json_extract(metadata_json, ?) IS NOT NULL
                  AND json_extract(metadata_json, ?) != ''
                ORDER BY val
                """,
                (f"$.{field}", document_id, f"$.{field}", f"$.{field}"),
            ).fetchall()
            values = [str(row["val"]) for row in rows if row["val"]]
            if values:
                result[field] = values
    return result


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
    delete_excel_policy_index(document_id)
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
