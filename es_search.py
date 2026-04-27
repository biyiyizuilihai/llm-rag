from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ES_URL = os.environ.get("ELASTICSEARCH_URL") or os.environ.get("OPENSEARCH_URL") or ""
ES_USERNAME = os.environ.get("ELASTICSEARCH_USERNAME") or os.environ.get("OPENSEARCH_USERNAME") or ""
ES_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD") or os.environ.get("OPENSEARCH_PASSWORD") or ""
ES_INDEX_PREFIX = os.environ.get("ELASTICSEARCH_INDEX_PREFIX", "llm_rag").strip() or "llm_rag"
ES_ANALYZER = os.environ.get("ELASTICSEARCH_CHINESE_ANALYZER", "ik_max_word").strip() or "ik_max_word"
ES_SEARCH_ANALYZER = os.environ.get("ELASTICSEARCH_CHINESE_SEARCH_ANALYZER", "ik_smart").strip() or "ik_smart"
ES_TIMEOUT = float(os.environ.get("ELASTICSEARCH_TIMEOUT", "5"))
ES_VERIFY_SSL = os.environ.get("ELASTICSEARCH_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}

PDF_INDEX = f"{ES_INDEX_PREFIX}_pdf_pages"
EXCEL_INDEX = f"{ES_INDEX_PREFIX}_excel_chunks"


def enabled() -> bool:
    return bool(ES_URL.strip())


def _url(path: str) -> str:
    return f"{ES_URL.rstrip('/')}/{path.lstrip('/')}"


def _auth() -> tuple[str, str] | None:
    if ES_USERNAME and ES_PASSWORD:
        return ES_USERNAME, ES_PASSWORD
    return None


def _request(method: str, path: str, **kwargs: Any) -> requests.Response:
    kwargs.setdefault("timeout", ES_TIMEOUT)
    kwargs.setdefault("verify", ES_VERIFY_SSL)
    auth = _auth()
    if auth:
        kwargs.setdefault("auth", auth)
    response = requests.request(method, _url(path), **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {response.status_code} {response.text[:500]}")
    return response


def _text_mapping(extra_properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "rag_zh_index": {"type": ES_ANALYZER},
                    "rag_zh_search": {"type": ES_SEARCH_ANALYZER},
                }
            }
        },
        "mappings": {
            "dynamic": True,
            "properties": {
                "document_id": {"type": "integer"},
                "file_name": {"type": "keyword"},
                "display_name": {
                    "type": "text",
                    "analyzer": "rag_zh_index",
                    "search_analyzer": "rag_zh_search",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "metadata_text": {
                    "type": "text",
                    "analyzer": "rag_zh_index",
                    "search_analyzer": "rag_zh_search",
                },
                **extra_properties,
            },
        },
    }


def _create_index(index_name: str, mapping: dict[str, Any]) -> None:
    exists = requests.head(_url(index_name), timeout=ES_TIMEOUT, verify=ES_VERIFY_SSL, auth=_auth())
    if exists.status_code == 200:
        return
    try:
        _request("PUT", index_name, json=mapping)
    except Exception as exc:
        if ES_ANALYZER == "standard" and ES_SEARCH_ANALYZER == "standard":
            raise
        logger.warning(
            "[es.index.create.fallback] index=%s analyzer=%s/%s error=%s",
            index_name,
            ES_ANALYZER,
            ES_SEARCH_ANALYZER,
            exc,
        )
        fallback = json.loads(json.dumps(mapping))
        fallback["settings"]["analysis"]["analyzer"]["rag_zh_index"] = {"type": "standard"}
        fallback["settings"]["analysis"]["analyzer"]["rag_zh_search"] = {"type": "standard"}
        _request("PUT", index_name, json=fallback)


def ensure_indices() -> None:
    if not enabled():
        return
    _create_index(
        PDF_INDEX,
        _text_mapping(
            {
                "page_number": {"type": "integer"},
                "ocr_text": {
                    "type": "text",
                    "analyzer": "rag_zh_index",
                    "search_analyzer": "rag_zh_search",
                },
            }
        ),
    )
    _create_index(
        EXCEL_INDEX,
        _text_mapping(
            {
                "chunk_id": {"type": "integer"},
                "policy_id": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "source_row": {"type": "integer"},
                "title": {
                    "type": "text",
                    "analyzer": "rag_zh_index",
                    "search_analyzer": "rag_zh_search",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "chunk_text": {
                    "type": "text",
                    "analyzer": "rag_zh_index",
                    "search_analyzer": "rag_zh_search",
                },
                "search_text": {
                    "type": "text",
                    "analyzer": "rag_zh_index",
                    "search_analyzer": "rag_zh_search",
                },
            }
        ),
    )


def delete_document(kind: str, document_id: int) -> None:
    if not enabled():
        return
    ensure_indices()
    index_name = EXCEL_INDEX if kind == "excel" else PDF_INDEX
    _request(
        "POST",
        f"{index_name}/_delete_by_query",
        json={"query": {"term": {"document_id": int(document_id)}}},
    )


def _bulk_index(index_name: str, docs: list[dict[str, Any]], id_field: str) -> int:
    if not docs:
        return 0
    lines: list[str] = []
    for doc in docs:
        lines.append(json.dumps({"index": {"_index": index_name, "_id": str(doc[id_field])}}, ensure_ascii=False))
        lines.append(json.dumps(doc, ensure_ascii=False))
    response = _request(
        "POST",
        "_bulk",
        data="\n".join(lines) + "\n",
        headers={"Content-Type": "application/x-ndjson"},
    )
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload.get("items", [])[:3], ensure_ascii=False))
    return len(docs)


def metadata_to_text(metadata: dict[str, Any]) -> str:
    return "\n".join(f"{key}：{value}" for key, value in metadata.items() if value is not None and str(value).strip())


def index_excel_chunks(chunks: list[dict[str, Any]]) -> int:
    if not enabled():
        return 0
    ensure_indices()
    docs = [
        {
            "chunk_id": int(chunk["chunk_id"]),
            "policy_id": int(chunk["policy_id"]),
            "document_id": int(chunk["document_id"]),
            "chunk_index": int(chunk["chunk_index"]),
            "chunk_text": chunk.get("chunk_text") or "",
            "search_text": chunk.get("search_text") or "",
            "title": chunk.get("title") or "",
            "source_row": int(chunk.get("source_row") or 0),
            "file_name": chunk.get("document_file_name") or "",
            "display_name": chunk.get("document_display_name") or "",
            "metadata": chunk.get("metadata") or {},
            "metadata_text": metadata_to_text(chunk.get("metadata") or {}),
        }
        for chunk in chunks
    ]
    return _bulk_index(EXCEL_INDEX, docs, "chunk_id")


def index_pdf_pages(document: dict[str, Any], page_text_map: dict[int, str]) -> int:
    if not enabled():
        return 0
    ensure_indices()
    document_id = int(document["id"])
    docs = [
        {
            "id": f"{document_id}:{page_number}",
            "document_id": document_id,
            "page_number": int(page_number),
            "ocr_text": text or "",
            "file_name": document.get("file_name") or "",
            "display_name": document.get("display_name") or "",
            "metadata_text": "\n".join(
                [
                    f"文件：{document.get('file_name') or ''}",
                    f"标题：{document.get('display_name') or ''}",
                    f"页码：{page_number}",
                ]
            ),
        }
        for page_number, text in page_text_map.items()
        if str(text or "").strip()
    ]
    return _bulk_index(PDF_INDEX, docs, "id")


def _filter_clauses(document_id: int | None, filters: dict[str, str] | None) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    if document_id is not None:
        clauses.append({"term": {"document_id": int(document_id)}})
    for key, value in (filters or {}).items():
        if str(value or "").strip():
            clauses.append({"match_phrase": {"metadata_text": f"{key}：{value}"}})
    return clauses


def search_excel_chunks(
    query: str,
    *,
    document_id: int | None = None,
    filters: dict[str, str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if not enabled() or not str(query or "").strip():
        return []
    ensure_indices()
    payload = {
        "size": int(top_k),
        "query": {
            "bool": {
                "filter": _filter_clauses(document_id, filters),
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^4", "metadata_text^3", "search_text^2", "chunk_text"],
                            "type": "best_fields",
                        }
                    }
                ],
            }
        },
    }
    response = _request("POST", f"{EXCEL_INDEX}/_search", json=payload).json()
    results: list[dict[str, Any]] = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source") or {}
        results.append(
            {
                "chunk_id": int(source["chunk_id"]),
                "policy_id": int(source["policy_id"]),
                "document_id": int(source["document_id"]),
                "document_file_name": source.get("file_name") or "",
                "document_display_name": source.get("display_name") or "",
                "title": source.get("title") or "",
                "source_row": int(source.get("source_row") or 0),
                "chunk_index": int(source.get("chunk_index") or 0),
                "chunk_text": source.get("chunk_text") or "",
                "search_text": source.get("search_text") or "",
                "metadata": source.get("metadata") or {},
                "rank": float(hit.get("_score") or 0),
                "retrieval_sources": ["elasticsearch"],
            }
        )
    return results


def search_pdf_pages(query: str, *, document_id: int, top_k: int = 8) -> list[int]:
    if not enabled() or not str(query or "").strip():
        return []
    ensure_indices()
    payload = {
        "size": int(top_k),
        "query": {
            "bool": {
                "filter": [{"term": {"document_id": int(document_id)}}],
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["display_name^3", "metadata_text^2", "ocr_text"],
                            "type": "best_fields",
                        }
                    }
                ],
            }
        },
    }
    response = _request("POST", f"{PDF_INDEX}/_search", json=payload).json()
    return [
        int((hit.get("_source") or {}).get("page_number"))
        for hit in response.get("hits", {}).get("hits", [])
        if (hit.get("_source") or {}).get("page_number") is not None
    ]
