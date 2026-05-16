from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
from pypdf import PdfReader, PdfWriter

from storage import (
    delete_page_ocr,
    fts_search_pages,
    get_chapter_summary,
    get_document,
    list_route_ready_documents,
    list_documents,
    get_page_ocr_text_map,
    phrase_search_pages,
    save_page_ocr,
    save_document_profile_vector,
    save_page_vector,
    update_chapter_summary,
    update_document_profile,
    update_ocr_status,
    vector_search_pages,
)

load_dotenv()

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-35b-a3b")
DEFAULT_BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL_CONTEXT_WINDOWS = {
    "qwen3.5-35b-a3b": 262_144,
}
UPLOAD_RENDER_DPI = 110
UPLOAD_RENDER_MAX_PIXELS = 900_000
UPLOAD_RENDER_JPEG_QUALITY = 72
LLM_RENDER_MAX_PIXELS = 640_000
LLM_RENDER_JPEG_QUALITY = 60
PADDLEOCR_JOB_URL = os.environ.get(
    "PADDLEOCR_JOB_URL",
    "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
)
PADDLEOCR_MODEL = os.environ.get("PADDLEOCR_MODEL", "PP-OCRv5")
PADDLEOCR_TOKEN = os.environ.get("PADDLEOCR_TOKEN", "")
SILICONFLOW_EMBEDDING_URL = os.environ.get(
    "SILICONFLOW_EMBEDDING_URL",
    "https://api.siliconflow.cn/v1/embeddings",
)
SILICONFLOW_EMBEDDING_MODEL = os.environ.get(
    "SILICONFLOW_EMBEDDING_MODEL",
    "BAAI/bge-m3",
)
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_TIMEOUT = 120
SILICONFLOW_BATCH_SIZE = max(
    1,
    int(os.environ.get("SILICONFLOW_EMBEDDING_BATCH_SIZE", "16")),
)
PADDLEOCR_POLL_INTERVAL = 5
PADDLEOCR_RETRY_ATTEMPTS = max(
    1,
    int(os.environ.get("PADDLEOCR_RETRY_ATTEMPTS", "3")),
)
PADDLEOCR_RETRY_BACKOFF = max(
    1,
    int(os.environ.get("PADDLEOCR_RETRY_BACKOFF", "5")),
)
PADDLEOCR_STALL_TIMEOUT = max(
    PADDLEOCR_POLL_INTERVAL * 2,
    int(os.environ.get("PADDLEOCR_STALL_TIMEOUT", "180")),
)
PADDLEOCR_JOB_TIMEOUT = max(
    PADDLEOCR_STALL_TIMEOUT,
    int(os.environ.get("PADDLEOCR_JOB_TIMEOUT", "1800")),
)
PADDLEOCR_CHUNK_PAGE_THRESHOLD = max(
    1,
    int(os.environ.get("PADDLEOCR_CHUNK_PAGE_THRESHOLD", "48")),
)
PADDLEOCR_CHUNK_SIZE = max(
    1,
    int(os.environ.get("PADDLEOCR_CHUNK_SIZE", "40")),
)
PADDLEOCR_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
PADDLEOCR_OPTIONAL_PAYLOAD = {
    "markdownIgnoreLabels": [],
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": False,
    "textDetLimitType": "min",
    "textDetLimitSideLen": 64,
    "textDetThresh": 0.3,
    "textDetBoxThresh": 0.6,
    "textDetUnclipRatio": 1.5,
    "textRecScoreThresh": 0,
    "parseLanguage": "default",
}
DEFAULT_SYSTEM_PROMPT = (
    "你是一名严谨的文档问答助手。"
    "回答时优先基于当前绑定文档。"
    "如果文档没有依据，不要编造。"
    "输出使用清晰的 Markdown。"
)
QUERY_UNDERSTANDING_SYSTEM_PROMPT = """You are a retrieval query planner for multilingual PDF search.
You only rewrite the user's question into a compact JSON retrieval plan.

Rules:
1. Do not answer the question.
2. Always output a single JSON object only.
3. Infer search-friendly variants, likely English terms, abbreviations, and attachment or section hints when helpful.
4. Prefer high-recall retrieval queries over conversational phrasing.
5. scope must be one of: targeted, broad, overview.
6. For all-caps acronyms, keep the original acronym and obvious punctuation variants, but do not invent expanded meanings unless the user provided them.

JSON schema:
{
  "normalized_query": string,
  "query_variants": [string],
  "aliases": [string],
  "keywords": [string],
  "likely_sections": [string],
  "scope": "targeted" | "broad" | "overview"
}
"""
logger = logging.getLogger("uvicorn.error").getChild("docqa.pdf")
CROSS_REFERENCE_PATTERN = re.compile(r"\b(?:Attachment|Appendix)\s+\d+[A-Z]?\b", re.IGNORECASE)
TITLE_REFERENCE_PATTERN = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9/&,\- ]{5,120}?)\s*\((?:Attachment|Appendix)\s+\d+[A-Z]?\)",
    re.IGNORECASE,
)
TOC_HINT_PATTERN = re.compile(
    r"\b(?:appendix\s+list|table\s+of\s+contents|contents)\b|\.{3,}",
    re.IGNORECASE,
)
REFERENCE_TARGET_HINT_PATTERN = re.compile(
    r"^\s*(?:SECTION\s+[A-Z0-9 ]*:?\s*)?(?:Attachment|Appendix)\s+\d+[A-Z]?\b",
    re.IGNORECASE,
)
SECTION_ANCHOR_PATTERN = re.compile(
    r"^\s*(?:SECTION\s+[A-Z0-9]+(?:[: ]|$)|\d+\.\d+\s+[A-Z][A-Za-z].{0,80})",
    re.IGNORECASE,
)
OVERVIEW_KEYWORDS = (
    "总结",
    "概述",
    "概览",
    "纵览",
    "整体",
    "全局",
    "全文",
    "核心内容",
    "按模块",
    "按章节",
    "梳理",
    "总览",
    "overview",
    "overall",
    "summary",
    "summarize",
    "high level",
)
BROAD_COVERAGE_KEYWORDS = (
    "所有",
    "全部",
    "汇总",
    "整理",
    "关键流程",
    "责任人",
    "输入输出",
    "时限",
    "阈值",
    "例外",
    "all ",
    "entire",
    "across the document",
    "by section",
    "by module",
)
GENERIC_QUERY_PREFIX_PATTERN = re.compile(
    r"^(?:请问|请帮我|帮我|麻烦|想知道|查一下|查查|查询|查看|帮忙|请|查)\s*",
    re.IGNORECASE,
)
GENERIC_QUERY_SUFFIX_PATTERN = re.compile(
    r"(?:的内容|内容是什么|是什么|有哪些|在哪个条款|在哪条|在哪个章节|在哪一条|在哪|怎么写|如何填写|怎么填)\s*$",
    re.IGNORECASE,
)
CJK_QUERY_CHUNK_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
RETRIEVAL_ALIAS_MAP = {
    "不合格报告": ["NCR", "NCR report", "Non Conformance Report", "non-conformance report"],
    "文件和记录": ["Document and Records", "document and records"],
    "on-hold卡片": ["ON-HOLD CARD", "On hold card", "Attachment A-1"],
    "隔离卡": ["ON-HOLD CARD", "Quarantine", "Attachment A-1"],
}
QUERY_UNDERSTANDING_MAX_TOKENS = 256
DOCUMENT_PROFILE_MAX_TOKENS = 384
DOCUMENT_CHAPTER_SUMMARY_MAX_TOKENS = max(
    4096,
    int(os.environ.get("DOCUMENT_CHAPTER_SUMMARY_MAX_TOKENS", "4096")),
)
DOCUMENT_CHAPTER_SUMMARY_CHUNK_SIZE = max(
    1,
    int(os.environ.get("DOCUMENT_CHAPTER_SUMMARY_CHUNK_SIZE", "40")),
)
DOCUMENT_ROUTING_TOP_K = 5
MULTI_DOC_TOTAL_PAGE_BUDGET = 15
MULTI_DOC_PER_DOC_PAGE_LIMIT = 6
MULTI_DOC_SINGLE_DOC_PAGE_LIMIT = 30
DOCUMENT_PROFILE_SYSTEM_PROMPT = """You are a document profiler for multilingual PDF routing.
Return a single JSON object only.

Rules:
1. Do not answer user questions.
2. Summarize what the document is about in one concise sentence.
3. Infer the document type.
4. Extract up to 12 routing keywords.
5. Extract up to 8 title aliases, abbreviations, or alternate names that could appear in user questions.

JSON schema:
{
  "summary_text": string,
  "doc_type": string,
  "keywords": [string],
  "title_aliases": [string]
}
"""
DOCUMENT_RELEVANCE_FILTER_PROMPT = """你是文档相关性判断助手。

系统已经先在多个文档中召回到了页面。请根据用户问题、文档名和文档画像摘要，判断哪些文档真正可能帮助回答问题。

判断标准：
- 文档必须包含问题涉及的实体、缩写、主题或流程，或者明显是同一业务领域资料。
- 宁可多选也不要漏选；但明显无关的文档不要选。
- 如果无法判断，保留可能相关的文档。

只返回 JSON，格式：{"relevant_document_ids": [1, 2]}
不要返回解释文字。"""
DOCUMENT_CHAPTER_SUMMARY_PROMPT = """你是一个文档内容分布分析助手。请根据以下 PDF 文档的完整 OCR 文本，输出“哪页到哪页主要讲什么”的内容分布摘要。

要求：
1. 识别主要内容块，可以是正式章节、节（Section）、附录（Appendix/Attachment）、表格/表单，也可以是自然主题段落。
2. 每个内容块单独一行，格式严格为：内容标题（第X-Y页）：一句话说明主要内容。
3. 按页码从小到大顺序输出。
4. 如果文档没有明显章节，不要硬编“第1章/第2章”，请按页面内容给出简短主题标题。
5. 如果文档只有一页或是表单/清单，也可以输出“整体内容（第1页）：...”或“表单主体（第1-2页）：...”。
6. 只输出内容分布列表，不要有任何其他说明或前缀。

示例输出：
引言与适用范围（第1-4页）：介绍项目背景、适用范围和主要术语定义。
质量要求与处理流程（第5-22页）：规定材料规格、工艺参数、检验标准及不合格品处理流程。
文件与记录要求（第23-28页）：说明文件控制要求、记录保存期限和变更管理程序。
检查记录表（第29-32页）：提供质量检查的标准记录表格和填写说明。
缩略词表（第33-34页）：列出文档中所用缩略词及其含义。"""
FORM_TEMPLATE_FIELD_HINTS = (
    "issue to",
    "lot no",
    "date/time",
    "problem description",
    "root cause",
    "corrective / preventive action",
    "failure category",
    "containment/immediate actions",
    "verification of corrective / preventive action",
)


def preview_text(text: str, limit: int = 160) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return {}

    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_string_list(value: Any, limit: int = 8) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = " ".join(str(item or "").split()).strip()
        key = normalized.casefold()
        if len(normalized) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if len(result) >= limit:
            break
    return result


def understand_retrieval_query(
    question: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    source = " ".join(str(question or "").split()).strip()
    if not source:
        return {}

    client = build_openai_client(base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Rewrite the following user question into a retrieval JSON plan. "
                    "Return JSON only.\n"
                    f"Question: {source}"
                ),
            },
        ],
        max_tokens=QUERY_UNDERSTANDING_MAX_TOKENS,
        temperature=0,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )

    raw_content = extract_response_text(response.choices[0].message.content)
    parsed = _extract_json_object(raw_content)
    if not parsed:
        logger.warning(
            "[query.understanding.invalid] question=%r raw=%r",
            preview_text(question),
            preview_text(raw_content, limit=240),
        )
        return {}

    plan = {
        "normalized_query": " ".join(str(parsed.get("normalized_query") or "").split()).strip(),
        "query_variants": _coerce_string_list(parsed.get("query_variants"), limit=8),
        "aliases": _coerce_string_list(parsed.get("aliases"), limit=8),
        "keywords": _coerce_string_list(parsed.get("keywords"), limit=8),
        "likely_sections": _coerce_string_list(parsed.get("likely_sections"), limit=6),
        "scope": str(parsed.get("scope") or "").strip().lower(),
    }
    logger.info(
        "[query.understanding] question=%r normalized=%r variants=%s aliases=%s keywords=%s sections=%s scope=%s",
        preview_text(question),
        plan["normalized_query"],
        plan["query_variants"],
        plan["aliases"],
        plan["keywords"],
        plan["likely_sections"],
        plan["scope"],
    )
    return plan


def build_query_variants(question: str, query_plan: dict[str, Any] | None = None) -> list[str]:
    source = " ".join(str(question or "").split())
    if not source:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        normalized = " ".join(str(candidate or "").split()).strip()
        key = normalized.casefold()
        if len(normalized) < 2 or key in seen:
            return
        seen.add(key)
        variants.append(normalized)

    add(source)

    if query_plan:
        add(query_plan.get("normalized_query") or "")
        for field in ("query_variants", "aliases", "keywords", "likely_sections"):
            for value in query_plan.get(field) or []:
                add(value)

    trimmed = GENERIC_QUERY_PREFIX_PATTERN.sub("", source)
    trimmed = GENERIC_QUERY_SUFFIX_PATTERN.sub("", trimmed).strip(" ：:，,。?？")
    add(trimmed)

    def add_cjk_fragments(value: str) -> None:
        for chunk in CJK_QUERY_CHUNK_PATTERN.findall(str(value or "")):
            add(chunk)
            for size in (4, 5, 6):
                if len(chunk) <= size:
                    continue
                for start in range(0, len(chunk) - size + 1):
                    add(chunk[start : start + size])

    add_cjk_fragments(source)
    add_cjk_fragments(trimmed)

    for term, aliases in RETRIEVAL_ALIAS_MAP.items():
        if term in source:
            add(term)
            for alias in aliases:
                add(alias)

    if trimmed and trimmed != source:
        for term, aliases in RETRIEVAL_ALIAS_MAP.items():
            if term in trimmed:
                for alias in aliases:
                    add(alias)

    if query_plan:
        for field in ("query_variants", "aliases", "keywords", "likely_sections"):
            for value in query_plan.get(field) or []:
                add_cjk_fragments(str(value or ""))

    return variants[:16]


def build_keyword_match_terms(question: str, query_plan: dict[str, Any] | None = None) -> list[str]:
    candidates = build_query_variants(question, query_plan=query_plan)
    seen: set[str] = set()
    ordered: list[str] = []

    def add(term: str) -> None:
        normalized = " ".join(str(term or "").split()).strip()
        if len(normalized) < 2:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        ordered.append(normalized)

    for candidate in candidates:
        add(candidate)
        for chunk in CJK_QUERY_CHUNK_PATTERN.findall(candidate):
            add(chunk)
            for size in (2, 3, 4, 5, 6):
                if len(chunk) <= size:
                    continue
                for start in range(0, len(chunk) - size + 1):
                    add(chunk[start : start + size])

    return ordered[:32]


def normalize_route_token(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())
    return normalized.strip()


def collect_document_profile_material(document: dict, text_map: dict[int, str]) -> str:
    total_pages = int(document.get("page_count") or 0)
    sampled_pages: list[int] = []
    if total_pages > 0:
        sampled_pages.extend(range(1, min(total_pages, 3) + 1))
        step = max(1, total_pages // 4)
        sampled_pages.extend(range(1, total_pages + 1, step))

    ordered_pages: list[int] = []
    seen: set[int] = set()
    for page_number in sampled_pages:
        if page_number in seen or page_number not in text_map:
            continue
        seen.add(page_number)
        ordered_pages.append(page_number)

    sampled_text_blocks = []
    for page_number in ordered_pages[:8]:
        text = preview_text(text_map.get(page_number, ""), limit=600)
        if not text:
            continue
        sampled_text_blocks.append(f"[Page {page_number}] {text}")

    combined_text = "\n".join(sampled_text_blocks)
    return (
        f"File name: {document.get('file_name', '')}\n"
        f"Display name: {document.get('display_name', '')}\n"
        f"Page count: {document.get('page_count', 0)}\n"
        f"Sampled OCR text:\n{combined_text}"
    ).strip()


def build_document_profile(document_id: int) -> dict[str, Any]:
    document = get_document(document_id)
    if not document:
        raise RuntimeError(f"文档不存在：{document_id}")

    text_map = get_page_ocr_text_map(document_id)
    material = collect_document_profile_material(document, text_map)
    if not material.strip():
        raise RuntimeError("OCR 文本为空，无法生成文档画像。")

    client = build_openai_client(DEFAULT_BASE_URL)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": DOCUMENT_PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": material},
        ],
        max_tokens=DOCUMENT_PROFILE_MAX_TOKENS,
        temperature=0,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )

    raw_content = extract_response_text(response.choices[0].message.content)
    parsed = _extract_json_object(raw_content)
    if not parsed:
        raise RuntimeError("文档画像生成失败：模型未返回有效 JSON。")

    summary_text = " ".join(str(parsed.get("summary_text") or "").split()).strip()
    doc_type = " ".join(str(parsed.get("doc_type") or "").split()).strip()
    keywords = _coerce_string_list(parsed.get("keywords"), limit=12)
    title_aliases = _coerce_string_list(parsed.get("title_aliases"), limit=8)

    if not summary_text:
        summary_text = preview_text(material, limit=240)
    route_parts = [
        str(document.get("file_name") or ""),
        str(document.get("display_name") or ""),
        summary_text,
        doc_type,
        " ".join(keywords),
        " ".join(title_aliases),
    ]
    route_text = "\n".join(part.strip() for part in route_parts if str(part).strip())

    update_document_profile(
        document_id,
        profile_status="done",
        profile_detail="文档画像已生成，可参与多文档路由。",
        summary_text=summary_text,
        doc_type=doc_type,
        keywords=keywords,
        title_aliases=title_aliases,
        route_text=route_text,
    )
    try:
        save_document_profile_vector(document_id, encode_text(route_text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[profile.vector.error] document_id=%s error=%s", document_id, exc)

    payload = get_document(document_id) or document
    logger.info(
        "[profile.done] document_id=%s doc_type=%r keywords=%s aliases=%s",
        document_id,
        doc_type,
        keywords,
        title_aliases,
    )
    return payload


def build_chapter_summary_user_message(
    document: dict[str, Any],
    text_map: dict[int, str],
    *,
    start_page: int,
    end_page: int,
) -> str:
    total_pages = int(document.get("page_count") or 0)
    page_blocks: list[str] = []
    for page_number in range(start_page, end_page + 1):
        text = str(text_map.get(page_number) or "").strip()
        page_blocks.append(f"[第{page_number}页]\n{text}" if text else f"[第{page_number}页]\n（空页）")

    return (
        f"文档：{document.get('file_name', '')}（共 {total_pages} 页）\n"
        f"当前需要分析的页码范围：第 {start_page}-{end_page} 页。\n"
        "请只输出这个页码范围内的内容分布，不要概括范围外页面。\n\n"
        + "\n\n".join(page_blocks)
    )


def create_chapter_summary_for_range(
    client: Any,
    document: dict[str, Any],
    text_map: dict[int, str],
    *,
    start_page: int,
    end_page: int,
) -> str:
    user_message = build_chapter_summary_user_message(
        document,
        text_map,
        start_page=start_page,
        end_page=end_page,
    )
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": DOCUMENT_CHAPTER_SUMMARY_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=DOCUMENT_CHAPTER_SUMMARY_MAX_TOKENS,
        temperature=0,
        extra_body={"enable_thinking": False},
    )

    chapter_summary = extract_response_text(response.choices[0].message.content).strip()
    if not chapter_summary:
        raise RuntimeError("章节摘要生成失败：模型未返回内容。")
    finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "").lower()
    if finish_reason == "length":
        raise RuntimeError(f"内容分布生成被截断：第 {start_page}-{end_page} 页输出达到 max_tokens。")
    return chapter_summary


def build_chapter_summary(document_id: int) -> str:
    document = get_document(document_id)
    if not document:
        raise RuntimeError(f"文档不存在：{document_id}")

    text_map = get_page_ocr_text_map(document_id)
    if not text_map:
        raise RuntimeError("OCR 文本为空，无法生成章节摘要。")

    total_pages = int(document.get("page_count") or 0)
    if total_pages <= 0:
        raise RuntimeError("PDF 页数为空，无法生成章节摘要。")

    client = build_openai_client(DEFAULT_BASE_URL)
    page_ranges = iter_pdf_chunk_ranges(total_pages, DOCUMENT_CHAPTER_SUMMARY_CHUNK_SIZE)
    summaries: list[str] = []
    for start_page, end_page in page_ranges:
        summaries.append(
            create_chapter_summary_for_range(
                client,
                document,
                text_map,
                start_page=start_page,
                end_page=end_page,
            )
        )

    chapter_summary = "\n".join(summary.strip() for summary in summaries if summary.strip()).strip()
    if not chapter_summary:
        raise RuntimeError("章节摘要生成失败：模型未返回内容。")

    update_chapter_summary(document_id, chapter_summary)
    logger.info(
        "[chapter_summary.done] document_id=%s chunks=%s lines=%s",
        document_id,
        len(page_ranges),
        len(chapter_summary.splitlines()),
    )
    return chapter_summary


def find_explicit_document_matches(question: str) -> list[dict[str, Any]]:
    normalized_question = normalize_route_token(question)
    if not normalized_question:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[int] = set()
    for document in list_documents():
        if str(document.get("ocr_status") or "") != "done":
            continue
        candidates = [
            str(document.get("file_name") or ""),
            str(document.get("display_name") or ""),
            *(document.get("title_aliases") or []),
        ]
        for candidate in candidates:
            normalized_candidate = normalize_route_token(candidate)
            if len(normalized_candidate) < 4:
                continue
            if normalized_candidate in normalized_question:
                document_id = int(document["id"])
                if document_id not in seen:
                    seen.add(document_id)
                    matches.append(document)
                break
    return matches


def route_documents(question: str, top_k: int = DOCUMENT_ROUTING_TOP_K) -> list[dict[str, Any]]:
    explicit_matches = find_explicit_document_matches(question)
    if explicit_matches:
        logger.info(
            "[doc.route.explicit] question=%r document_ids=%s",
            preview_text(question),
            [document["id"] for document in explicit_matches],
        )
        return explicit_matches[:top_k]

    route_ready_docs = list_route_ready_documents()
    if not route_ready_docs:
        route_ready_docs = [
            document
            for document in list_documents()
            if str(document.get("ocr_status") or "").strip() == "done"
        ]
        logger.warning(
            "[doc.route.fallback_no_profile] question=%r document_ids=%s",
            preview_text(question),
            [int(document["id"]) for document in route_ready_docs],
        )

    logger.info(
        "[doc.route] question=%r document_ids=%s",
        preview_text(question),
        [int(document["id"]) for document in route_ready_docs],
    )
    return route_ready_docs


def is_toc_like_page(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False

    references = {match.upper() for match in CROSS_REFERENCE_PATTERN.findall(cleaned)}
    reference_count = len(CROSS_REFERENCE_PATTERN.findall(cleaned))
    if TOC_HINT_PATTERN.search(cleaned):
        return True
    if cleaned.lower().startswith("no attachment title"):
        return True
    if len(references) >= 4:
        return True
    if cleaned.count("Appendix") + cleaned.count("Attachment") >= 6:
        return True
    if reference_count >= 3 and len(cleaned) <= 260:
        return True
    return False


def is_reference_target_page(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned or is_toc_like_page(cleaned):
        return False
    return bool(REFERENCE_TARGET_HINT_PATTERN.search(cleaned[:220]))


def is_section_anchor_page(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned or is_toc_like_page(cleaned) or is_reference_target_page(cleaned):
        return False
    return bool(SECTION_ANCHOR_PATTERN.search(cleaned[:220]))


def is_form_template_page(text: str) -> bool:
    cleaned = " ".join(str(text or "").split()).lower()
    if not cleaned or is_toc_like_page(cleaned):
        return False

    hint_count = sum(1 for hint in FORM_TEMPLATE_FIELD_HINTS if hint in cleaned)
    if hint_count >= 3:
        return True

    return False


def detect_question_scope(question: str) -> str:
    lowered = str(question or "").strip().lower()
    if not lowered:
        return "targeted"

    if any(keyword.lower() in lowered for keyword in OVERVIEW_KEYWORDS):
        return "overview"
    if any(keyword.lower() in lowered for keyword in BROAD_COVERAGE_KEYWORDS):
        return "broad"
    return "targeted"


def is_content_focused_question(question: str, query_plan: dict[str, Any] | None = None) -> bool:
    joined = " ".join(
        [
            str(question or ""),
            str((query_plan or {}).get("normalized_query") or ""),
            " ".join((query_plan or {}).get("query_variants") or []),
            " ".join((query_plan or {}).get("keywords") or []),
        ]
    ).lower()
    hints = ("内容", "表单", "格式", "模板", "content", "form", "report", "template")
    return any(hint in joined for hint in hints)


def score_keyword_match_pages(
    question: str,
    query_plan: dict[str, Any] | None,
    page_text_map: dict[int, str],
    *,
    top_k: int = 8,
) -> list[int]:
    terms = build_keyword_match_terms(question, query_plan=query_plan)
    if not terms or not page_text_map:
        return []

    ranked: list[tuple[int, int]] = []
    source_chunks = {
        chunk
        for chunk in CJK_QUERY_CHUNK_PATTERN.findall(str(question or ""))
        if len(chunk) >= 2
    }
    folded_source_chunks = [chunk.casefold() for chunk in source_chunks]
    for page_number, page_text in page_text_map.items():
        text = str(page_text or "")
        if not text:
            continue

        folded_text = text.casefold()
        matched_terms = [term for term in terms if term.casefold() in folded_text]
        if not matched_terms:
            continue

        score = 0
        unique_matches: list[str] = []
        seen_matches: set[str] = set()
        source_hit_count = 0
        for term in matched_terms:
            key = term.casefold()
            if key in seen_matches:
                continue
            seen_matches.add(key)
            unique_matches.append(term)
            score += min(6, max(1, len(term)))
            if any(key in chunk or chunk in key for chunk in folded_source_chunks):
                source_hit_count += 1

        score += min(8, len(unique_matches) * 2)
        if source_hit_count >= 2:
            score += 6
        elif source_hit_count == 1:
            score += 2
        if any(len(term) >= 6 for term in unique_matches):
            score += 2

        ranked.append((page_number, score))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [page_number for page_number, _ in ranked[:top_k]]


def collect_overview_pages(
    document_id: int,
    total_pages: int,
    max_pages: int,
) -> list[int]:
    text_map = get_page_ocr_text_map(document_id)
    ordered: list[int] = []
    seen: set[int] = set()

    def append_page(page_number: int) -> None:
        if not (1 <= page_number <= total_pages):
            return
        if page_number in seen:
            return
        if is_toc_like_page(text_map.get(page_number, "")):
            return
        seen.add(page_number)
        ordered.append(page_number)

    for page_number in range(1, min(total_pages, 4) + 1):
        append_page(page_number)

    for page_number in range(1, total_pages + 1):
        text = text_map.get(page_number, "")
        if is_section_anchor_page(text) or is_reference_target_page(text):
            append_page(page_number)
        if len(ordered) >= max_pages:
            break

    if len(ordered) < max_pages:
        step = max(1, total_pages // max_pages)
        for page_number in range(1, total_pages + 1, step):
            append_page(page_number)
            if len(ordered) >= max_pages:
                break

    return ordered[:max_pages]


def collect_sparse_fallback_pages(
    document_id: int,
    total_pages: int,
    max_pages: int,
) -> list[int]:
    text_map = get_page_ocr_text_map(document_id)
    anchor_pages = [
        page_number
        for page_number in range(1, total_pages + 1)
        if not is_toc_like_page(text_map.get(page_number, ""))
        and (is_section_anchor_page(text_map.get(page_number, "")) or is_reference_target_page(text_map.get(page_number, "")))
    ]

    if not anchor_pages:
        return collect_overview_pages(document_id, total_pages, max_pages)

    seed_budget = max(1, min(len(anchor_pages), max_pages // 2 or 1))
    if len(anchor_pages) <= seed_budget:
        seed_pages = anchor_pages
    else:
        step = (len(anchor_pages) - 1) / max(seed_budget - 1, 1)
        seed_pages = []
        seen_seed: set[int] = set()
        for index in range(seed_budget):
            page_number = anchor_pages[round(index * step)]
            if page_number in seen_seed:
                continue
            seen_seed.add(page_number)
            seed_pages.append(page_number)

    ordered: list[int] = []
    seen_pages: set[int] = set()

    def append_page(page_number: int) -> None:
        if not (1 <= page_number <= total_pages):
            return
        if page_number in seen_pages:
            return
        if is_toc_like_page(text_map.get(page_number, "")):
            return
        seen_pages.add(page_number)
        ordered.append(page_number)

    for page_number in seed_pages:
        append_page(page_number)
        append_page(page_number + 1)
        if len(ordered) >= max_pages:
            return ordered[:max_pages]

    if len(ordered) < max_pages:
        step = max(1, total_pages // max_pages)
        for page_number in range(1, total_pages + 1, step):
            append_page(page_number)
            if len(ordered) >= max_pages:
                break

    return ordered[:max_pages]


def extract_cross_reference_queries(text: str) -> list[str]:
    seen: set[str] = set()
    queries: list[str] = []
    source = str(text or "")

    for match in TITLE_REFERENCE_PATTERN.findall(source):
        normalized = " ".join(match.split())
        normalized = re.sub(r"^[0-9]+(?:\.[0-9]+)*[.):\-]?\s+", "", normalized).strip(" :-")
        key = normalized.upper()
        if len(normalized) < 8 or key in seen:
            continue
        seen.add(key)
        queries.append(normalized)

    for match in CROSS_REFERENCE_PATTERN.findall(source):
        normalized = " ".join(match.split())
        key = normalized.upper()
        if key in seen:
            continue
        seen.add(key)
        queries.append(normalized)
    return queries


def expand_cross_reference_pages(
    document_id: int,
    seed_pages: list[int],
) -> tuple[list[str], list[int]]:
    if not seed_pages:
        return [], []

    seed_text_map = get_page_ocr_text_map(document_id, seed_pages)
    queries: list[str] = []
    seen_queries: set[str] = set()
    for page_number in seed_pages:
        for query in extract_cross_reference_queries(seed_text_map.get(page_number, "")):
            key = query.upper()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            queries.append(query)

    if not queries:
        return [], []

    collected: list[int] = []
    seen_pages: set[int] = set(seed_pages)
    for query in queries[:8]:
        if " " in query and not query.lower().startswith(("attachment ", "appendix ")):
            query_hits = phrase_search_pages(query, document_id, top_k=10)
        else:
            query_hits = fts_search_pages(query, document_id, top_k=8)
        if not query_hits:
            continue

        text_map = get_page_ocr_text_map(document_id, query_hits)
        target_hits = [page for page in query_hits if is_reference_target_page(text_map.get(page, ""))]
        non_toc_hits = [page for page in query_hits if not is_toc_like_page(text_map.get(page, ""))]
        effective_hits = target_hits or non_toc_hits or query_hits

        distant_hits = [
            page
            for page in effective_hits
            if all(abs(page - seed_page) > 2 for seed_page in seed_pages)
        ]
        ordered_hits = distant_hits + [page for page in effective_hits if page not in distant_hits]

        for page in ordered_hits[:2]:
            if page in seen_pages:
                continue
            seen_pages.add(page)
            collected.append(page)

    return queries[:8], collected


def resize_image_if_needed(img: Image.Image, max_pixels: Optional[int]) -> Image.Image:
    if not max_pixels:
        return img

    w, h = img.size
    pixels = w * h
    if pixels <= max_pixels:
        return img

    scale = math.sqrt(max_pixels / pixels)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def pil_to_jpeg_bytes(img: Image.Image, jpeg_quality: int = 80) -> bytes:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return buf.getvalue()


def pil_to_data_url(img: Image.Image, jpeg_quality: int = 80) -> str:
    jpeg_bytes = pil_to_jpeg_bytes(img, jpeg_quality=jpeg_quality)
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def render_pdf_pages(
    pdf_path: str | Path,
    start_page: int = 1,
    end_page: Optional[int] = None,
    dpi: int = UPLOAD_RENDER_DPI,
    local_max_pixels: Optional[int] = UPLOAD_RENDER_MAX_PIXELS,
    jpeg_quality: int = UPLOAD_RENDER_JPEG_QUALITY,
) -> list[tuple[int, str]]:
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if end_page is None:
        end_page = total_pages

    if start_page < 1 or end_page > total_pages or start_page > end_page:
        raise ValueError(
            f"页码范围无效：start_page={start_page}, end_page={end_page}, total_pages={total_pages}"
        )

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    result: list[tuple[int, str]] = []
    for page_num in range(start_page, end_page + 1):
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = resize_image_if_needed(img, local_max_pixels)
        data_url = pil_to_data_url(img, jpeg_quality=jpeg_quality)
        result.append((page_num, data_url))

    doc.close()
    return result


def render_pdf_to_jpegs(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = UPLOAD_RENDER_DPI,
    local_max_pixels: Optional[int] = UPLOAD_RENDER_MAX_PIXELS,
    jpeg_quality: int = UPLOAD_RENDER_JPEG_QUALITY,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    result: list[Path] = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = resize_image_if_needed(img, local_max_pixels)
        jpg_bytes = pil_to_jpeg_bytes(img, jpeg_quality=jpeg_quality)
        target = output_path / f"page-{page_index + 1:04d}.jpg"
        target.write_bytes(jpg_bytes)
        result.append(target)

    doc.close()
    return result


def get_llm_image_cache_dir(
    render_dir: str | Path,
    max_pixels: Optional[int],
    jpeg_quality: int,
) -> Path:
    pixel_label = str(max_pixels) if max_pixels else "orig"
    return Path(render_dir) / f"_llm_{pixel_label}_{jpeg_quality}"


def ensure_llm_optimized_jpeg(
    source_path: Path,
    cache_dir: Path,
    max_pixels: Optional[int],
    jpeg_quality: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / source_path.name
    if target.exists() and target.stat().st_mtime >= source_path.stat().st_mtime:
        return target

    with Image.open(source_path) as img:
        img.load()
        optimized = resize_image_if_needed(img, max_pixels)
        jpg_bytes = pil_to_jpeg_bytes(optimized, jpeg_quality=jpeg_quality)

    target.write_bytes(jpg_bytes)
    return target


def load_rendered_page_data_urls(
    render_dir: str | Path,
    max_pixels: Optional[int] = LLM_RENDER_MAX_PIXELS,
    jpeg_quality: int = LLM_RENDER_JPEG_QUALITY,
) -> list[tuple[int, str]]:
    render_path = Path(render_dir)
    cache_dir = get_llm_image_cache_dir(render_path, max_pixels, jpeg_quality)
    page_urls: list[tuple[int, str]] = []
    for image_path in sorted(render_path.glob("page-*.jpg")):
        try:
            page_num = int(image_path.stem.split("-")[-1])
        except ValueError:
            continue

        optimized_path = ensure_llm_optimized_jpeg(
            source_path=image_path,
            cache_dir=cache_dir,
            max_pixels=max_pixels,
            jpeg_quality=jpeg_quality,
        )
        b64 = base64.b64encode(optimized_path.read_bytes()).decode("utf-8")
        page_urls.append((page_num, f"data:image/jpeg;base64,{b64}"))
    return page_urls


def load_selected_rendered_page_data_urls(
    render_dir: str | Path,
    page_numbers: list[int],
    max_pixels: Optional[int] = LLM_RENDER_MAX_PIXELS,
    jpeg_quality: int = LLM_RENDER_JPEG_QUALITY,
) -> list[tuple[int, str]]:
    render_path = Path(render_dir)
    cache_dir = get_llm_image_cache_dir(render_path, max_pixels, jpeg_quality)
    result: list[tuple[int, str]] = []
    for page_num in page_numbers:
        image_path = render_path / f"page-{page_num:04d}.jpg"
        if not image_path.exists():
            continue
        optimized_path = ensure_llm_optimized_jpeg(
            source_path=image_path,
            cache_dir=cache_dir,
            max_pixels=max_pixels,
            jpeg_quality=jpeg_quality,
        )
        b64 = base64.b64encode(optimized_path.read_bytes()).decode("utf-8")
        result.append((page_num, f"data:image/jpeg;base64,{b64}"))
    return result


def make_page_image_items(
    document: dict,
    page_images: list[tuple[int, str]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for page_num, data_url in page_images:
        items.append(
            {
                "document_id": int(document["id"]),
                "document_name": document.get("file_name") or "",
                "document_display_name": document.get("display_name") or document.get("file_name") or "",
                "page_number": int(page_num),
                "data_url": data_url,
                "pdf_url": document.get("pdf_url") or "",
            }
        )
    return items


def build_document_user_message(
    page_images: list[dict[str, Any]],
    question: str,
    api_max_pixels_per_page: Optional[int] = LLM_RENDER_MAX_PIXELS,
    document_summaries: dict[int, str] | None = None,
) -> dict[str, Any]:
    document_ids = {int(item.get("document_id") or 0) for item in page_images if item.get("document_id")}
    if len(document_ids) > 1:
        scope_intro = "你将看到多个 PDF 的页面图片。这些页面来自系统为当前问题召回出的候选文档。"
    else:
        scope_intro = "你将看到同一个 PDF 的页面图片。这是当前会话唯一需要参考的文档。"
    intro = scope_intro + (
        "请基于这些页面回答问题。"
        "如果答案依赖具体页面，请尽量注明页码。"
        "若信息不足，请明确说明缺少哪部分。"
        "\n\n"
        "**页码说明**：每条 '--- 文档名 · 第 N 页 ---' 中的 N 是该页在 PDF 文件中的物理页序号。"
        "引用页码时必须使用这个物理页号 N，不要使用文档内部印刷的页码数字。"
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
    if document_summaries:
        summary_lines = ["【文档内容分布参考】", "（以下为各文档的页码范围摘要，供你定位答案所在页面，实际内容以下方页面图片为准）", ""]
        seen_doc_ids: set[int] = set()
        for item in page_images:
            doc_id = int(item.get("document_id") or 0)
            if doc_id in seen_doc_ids or doc_id not in document_summaries:
                continue
            seen_doc_ids.add(doc_id)
            document_name = str(item.get("document_display_name") or item.get("document_name") or "文档")
            summary = str(document_summaries.get(doc_id) or "").strip()
            if not summary:
                continue
            summary_lines.append(f"▶ {document_name}")
            summary_lines.append(summary)
            summary_lines.append("")
        if seen_doc_ids:
            content.append({"type": "text", "text": "\n".join(summary_lines)})

    for item in page_images:
        page_num = int(item["page_number"])
        data_url = str(item["data_url"])
        document_name = str(item.get("document_display_name") or item.get("document_name") or "文档")
        content.append({"type": "text", "text": f"--- {document_name} · 第 {page_num} 页 ---"})
        image_item: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        if api_max_pixels_per_page:
            image_item["max_pixels"] = api_max_pixels_per_page
        content.append(image_item)

    content.append({"type": "text", "text": f"用户问题：{question}"})
    return {"role": "user", "content": content}


def limit_history_with_anchor(
    conversation_history: list[dict[str, Any]],
    max_history_messages: int,
) -> list[dict[str, Any]]:
    if not conversation_history or len(conversation_history) <= max_history_messages:
        return conversation_history

    anchor = conversation_history[0]
    tail_size = max(0, max_history_messages - 1)
    if tail_size == 0:
        return [anchor]
    return [anchor, *conversation_history[-tail_size:]]


def build_model_history(
    conversation_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_history: list[dict[str, Any]] = []
    for item in conversation_history:
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        model_history.append({"role": role, "content": item.get("content", "")})
    return model_history


def build_chat_messages(
    page_images: list[dict[str, Any]],
    question: str,
    conversation_history: list[dict[str, Any]],
    api_max_pixels_per_page: Optional[int] = LLM_RENDER_MAX_PIXELS,
    max_history_messages: int = 12,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    document_summaries: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    if conversation_history:
        history = build_model_history(conversation_history=conversation_history)
        messages.extend(limit_history_with_anchor(history, max_history_messages))

    messages.append(
        build_document_user_message(
            page_images=page_images,
            question=question,
            api_max_pixels_per_page=api_max_pixels_per_page,
            document_summaries=document_summaries,
        )
    )
    return messages


def build_openai_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请先设置环境变量 LLM_API_KEY 或 DASHSCOPE_API_KEY")
    logger.info(
        "[llm] create client base_url=%s api_key_len=%s",
        base_url,
        len(api_key),
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def usage_to_dict(usage_obj: Any) -> dict[str, Any]:
    if usage_obj is None:
        return {}
    if isinstance(usage_obj, dict):
        return usage_obj
    if hasattr(usage_obj, "model_dump"):
        return usage_obj.model_dump()
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
        "completion_tokens": getattr(usage_obj, "completion_tokens", None),
        "total_tokens": getattr(usage_obj, "total_tokens", None),
        "prompt_tokens_details": getattr(usage_obj, "prompt_tokens_details", None),
    }


def extract_response_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, dict):
        if message_content.get("type") == "text":
            return message_content.get("text", "")
        for key in ("text", "content", "value"):
            if isinstance(message_content.get(key), str):
                return message_content[key]
        return ""
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
        return "".join(parts).strip()
    return ""


def extract_delta_text(delta: Any) -> str:
    if delta is None:
        return ""

    content_text = extract_response_text(getattr(delta, "content", None))
    if content_text:
        return content_text

    model_extra = getattr(delta, "model_extra", None)
    if isinstance(model_extra, dict):
        return extract_response_text(model_extra.get("content"))

    return ""


def extract_delta_reasoning(delta: Any) -> str:
    if delta is None:
        return ""

    candidates = [
        getattr(delta, "reasoning_content", None),
        getattr(delta, "reasoning", None),
        getattr(delta, "thinking", None),
        getattr(delta, "reasoning_text", None),
    ]

    model_extra = getattr(delta, "model_extra", None)
    if isinstance(model_extra, dict):
        candidates.extend(
            [
                model_extra.get("reasoning_content"),
                model_extra.get("reasoning"),
                model_extra.get("thinking"),
                model_extra.get("reasoning_text"),
            ]
        )

    for candidate in candidates:
        text = extract_response_text(candidate)
        if text:
            return text

    return ""


def build_summary(
    model: str,
    page_images: list[dict[str, Any]],
    usage: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "context_window": MODEL_CONTEXT_WINDOWS.get(model),
        "page_count_sent": len(page_images),
        "pages_sent": [
            {
                "document_id": item.get("document_id"),
                "document_name": item.get("document_display_name") or item.get("document_name"),
                "page_number": item.get("page_number"),
            }
            for item in page_images
        ],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "requested_max_tokens": max_tokens,
    }


def submit_ocr_job(pdf_path: str | Path) -> str:
    if not PADDLEOCR_TOKEN:
        raise EnvironmentError("请先设置环境变量 PADDLEOCR_TOKEN")

    headers = {
        "Authorization": f"bearer {PADDLEOCR_TOKEN}",
    }
    data = {
        "model": PADDLEOCR_MODEL,
        "optionalPayload": json.dumps(PADDLEOCR_OPTIONAL_PAYLOAD),
    }

    with open(pdf_path, "rb") as pdf_file:
        response = requests.post(
            PADDLEOCR_JOB_URL,
            headers=headers,
            data=data,
            files={"file": pdf_file},
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("data", {}).get("jobId")
    if not job_id:
        raise RuntimeError(f"OCR 任务提交失败：{payload}")
    logger.info("[ocr] submitted job_id=%s pdf=%s", job_id, pdf_path)
    return str(job_id)


def poll_ocr_job(
    job_id: str,
    progress_callback: Any | None = None,
) -> str:
    if not PADDLEOCR_TOKEN:
        raise EnvironmentError("请先设置环境变量 PADDLEOCR_TOKEN")

    headers = {
        "Authorization": f"bearer {PADDLEOCR_TOKEN}",
    }
    started_at = time.monotonic()
    last_progress_at = started_at
    last_extracted_pages = -1

    while True:
        now = time.monotonic()
        if now - started_at > PADDLEOCR_JOB_TIMEOUT:
            raise RuntimeError(
                f"OCR 任务超时（超过 {PADDLEOCR_JOB_TIMEOUT} 秒），请重试重建索引。"
            )

        response = requests.get(
            f"{PADDLEOCR_JOB_URL}/{job_id}",
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json().get("data", {})
        state = payload.get("state")
        extract_progress = payload.get("extractProgress") or {}
        extracted_pages = int(extract_progress.get("extractedPages") or 0)
        total_pages = int(extract_progress.get("totalPages") or 0)
        if extracted_pages > last_extracted_pages:
            last_extracted_pages = extracted_pages
            last_progress_at = time.monotonic()
        if progress_callback:
            progress_callback(
                state=state,
                extracted_pages=extracted_pages,
                total_pages=total_pages,
            )
        if (
            state == "running"
            and extracted_pages >= 0
            and time.monotonic() - last_progress_at > PADDLEOCR_STALL_TIMEOUT
        ):
            raise RuntimeError(
                "OCR 任务长时间无进展，已自动终止。"
                f"当前停留在 {extracted_pages}/{total_pages or '?'} 页，请重试重建索引。"
            )
        if state == "done":
            jsonl_url = payload.get("resultUrl", {}).get("jsonUrl")
            if not jsonl_url:
                raise RuntimeError("OCR 任务完成，但缺少 jsonUrl")
            logger.info(
                "[ocr] job_id=%s done extracted_pages=%s total_pages=%s",
                job_id,
                extracted_pages,
                total_pages,
            )
            jsonl_response = requests.get(jsonl_url, timeout=120)
            jsonl_response.raise_for_status()
            return jsonl_response.text
        if state == "failed":
            raise RuntimeError(payload.get("errorMsg") or "OCR 任务失败")
        logger.info(
            "[ocr] job_id=%s state=%s extracted_pages=%s total_pages=%s",
            job_id,
            state,
            extracted_pages,
            total_pages,
        )
        time.sleep(PADDLEOCR_POLL_INTERVAL)


def extract_status_code_from_error(message: str) -> int | None:
    match = re.search(r"\b(?:status|状态)码\s*(\d{3})\b", message, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(429|500|502|503|504)\b", message)
    if match:
        return int(match.group(1))
    return None


def is_retryable_ocr_error(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        status_code = getattr(exc.response, "status_code", None)
        return int(status_code or 0) in PADDLEOCR_RETRYABLE_STATUS_CODES
    if isinstance(exc, requests.RequestException):
        return True
    status_code = extract_status_code_from_error(str(exc))
    return int(status_code or 0) in PADDLEOCR_RETRYABLE_STATUS_CODES


def iter_pdf_chunk_ranges(total_pages: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for start_page in range(1, total_pages + 1, chunk_size):
        end_page = min(total_pages, start_page + chunk_size - 1)
        ranges.append((start_page, end_page))
    return ranges


def write_pdf_chunk(
    reader: PdfReader,
    output_path: str | Path,
    start_page: int,
    end_page: int,
) -> None:
    writer = PdfWriter()
    for page_index in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_index])
    with open(output_path, "wb") as chunk_file:
        writer.write(chunk_file)


def run_ocr_job_with_retries(
    pdf_path: str | Path,
    *,
    progress_callback: Any | None = None,
    retry_callback: Any | None = None,
    job_label: str = "整份文档",
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, PADDLEOCR_RETRY_ATTEMPTS + 1):
        try:
            job_id = submit_ocr_job(pdf_path)
            return poll_ocr_job(job_id, progress_callback=progress_callback)
        except Exception as exc:
            last_error = exc
            if attempt >= PADDLEOCR_RETRY_ATTEMPTS or not is_retryable_ocr_error(exc):
                raise
            delay = PADDLEOCR_RETRY_BACKOFF * attempt
            logger.warning(
                "[ocr.retry] label=%s attempt=%s/%s delay=%ss pdf=%s error=%s",
                job_label,
                attempt,
                PADDLEOCR_RETRY_ATTEMPTS,
                delay,
                pdf_path,
                exc,
            )
            if retry_callback:
                retry_callback(
                    attempt=attempt,
                    max_attempts=PADDLEOCR_RETRY_ATTEMPTS,
                    exc=exc,
                    delay=delay,
                    job_label=job_label,
                )
            time.sleep(delay)
    if last_error:
        raise last_error
    raise RuntimeError("OCR 重试未返回结果。")


def collect_ocr_pages(
    pdf_path: str | Path,
    total_pages: int,
    *,
    progress_callback: Any | None = None,
    retry_callback: Any | None = None,
) -> list[dict[str, Any]]:
    if total_pages <= PADDLEOCR_CHUNK_PAGE_THRESHOLD:
        jsonl_text = run_ocr_job_with_retries(
            pdf_path,
            progress_callback=progress_callback,
            retry_callback=retry_callback,
            job_label=f"整份文档({total_pages}页)",
        )
        return parse_ocr_jsonl(jsonl_text, expected_pages=total_pages)

    reader = PdfReader(str(pdf_path))
    chunk_ranges = iter_pdf_chunk_ranges(total_pages, PADDLEOCR_CHUNK_SIZE)
    pages: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="ocr-chunks-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for chunk_index, (start_page, end_page) in enumerate(chunk_ranges, start=1):
            chunk_total = end_page - start_page + 1
            chunk_path = temp_dir_path / f"chunk-{chunk_index:03d}-{start_page}-{end_page}.pdf"
            write_pdf_chunk(reader, chunk_path, start_page, end_page)

            def chunk_progress_callback(*, state: str | None, extracted_pages: int, total_pages: int) -> None:
                if not progress_callback:
                    return
                global_total = len(reader.pages)
                effective_total = total_pages or chunk_total
                global_extracted = start_page - 1 + min(max(extracted_pages, 0), effective_total)
                progress_callback(
                    state=state,
                    extracted_pages=global_extracted,
                    total_pages=global_total,
                )

            chunk_label = (
                f"分段 {chunk_index}/{len(chunk_ranges)}"
                f"（第 {start_page}-{end_page} 页）"
            )
            jsonl_text = run_ocr_job_with_retries(
                chunk_path,
                progress_callback=chunk_progress_callback,
                retry_callback=retry_callback,
                job_label=chunk_label,
            )
            chunk_pages = parse_ocr_jsonl(jsonl_text, expected_pages=chunk_total)
            for local_offset, page in enumerate(chunk_pages, start=start_page):
                pages.append(
                    {
                        "page_number": local_offset,
                        "text": str(page.get("text") or ""),
                    }
                )
            logger.info(
                "[ocr.chunk.done] chunk=%s/%s pages=%s-%s total_pages=%s pdf=%s",
                chunk_index,
                len(chunk_ranges),
                start_page,
                end_page,
                len(reader.pages),
                pdf_path,
            )

    if len(pages) != total_pages:
        raise ValueError(f"OCR 分段结果页数异常：得到 {len(pages)} 页，预期 {total_pages} 页")
    return pages


def parse_ocr_jsonl(jsonl_text: str, expected_pages: int | None = None) -> list[dict[str, Any]]:
    lines = jsonl_text.strip().splitlines()
    pages: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        result = json.loads(line).get("result", {})
        ocr_results = result.get("ocrResults", [])
        if not ocr_results:
            pages.append({"page_number": len(pages) + 1, "text": ""})
            continue

        for ocr_result in ocr_results:
            texts: list[str] = []
            pruned_result = ocr_result.get("prunedResult") or {}

            rec_texts = pruned_result.get("rec_texts")
            if isinstance(rec_texts, list):
                for item in rec_texts:
                    text = str(item or "").strip()
                    if text:
                        texts.append(text)

            if not texts:
                for block in ocr_result.get("textBlocks", []):
                    text = str(block.get("text", "")).strip()
                    if text:
                        texts.append(text)

            pages.append(
                {
                    "page_number": len(pages) + 1,
                    "text": "\n".join(texts),
                }
            )

    if expected_pages is not None and len(pages) != expected_pages:
        raise ValueError(
            f"OCR 页数校验失败：OCR 返回 {len(pages)} 页，PDF 实际 {expected_pages} 页"
        )
    return pages


def encode_texts(texts: list[str]) -> list[list[float]]:
    if not SILICONFLOW_API_KEY:
        raise EnvironmentError("请先设置环境变量 SILICONFLOW_API_KEY")
    if not texts:
        return []

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    embeddings: list[list[float]] = []
    logger.info(
        "[embedding] request batches=%s items=%s model=%s",
        math.ceil(len(texts) / SILICONFLOW_BATCH_SIZE),
        len(texts),
        SILICONFLOW_EMBEDDING_MODEL,
    )

    for start in range(0, len(texts), SILICONFLOW_BATCH_SIZE):
        chunk = texts[start : start + SILICONFLOW_BATCH_SIZE]
        payload = {
            "model": SILICONFLOW_EMBEDDING_MODEL,
            "input": chunk if len(chunk) > 1 else chunk[0],
        }
        response = requests.post(
            SILICONFLOW_EMBEDDING_URL,
            json=payload,
            headers=headers,
            timeout=SILICONFLOW_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        data = result.get("data") or []
        if not data:
            raise RuntimeError(f"SiliconFlow embedding 响应无效：{result}")

        sorted_items = sorted(data, key=lambda item: int(item.get("index", 0)))
        chunk_embeddings = [list(item.get("embedding") or []) for item in sorted_items]
        if len(chunk_embeddings) != len(chunk):
            raise RuntimeError(
                f"SiliconFlow embedding 返回数量异常：请求 {len(chunk)} 条，返回 {len(chunk_embeddings)} 条"
            )
        embeddings.extend(chunk_embeddings)
        logger.info(
            "[embedding] batch ok start=%s size=%s first_text=%r",
            start,
            len(chunk),
            preview_text(chunk[0]) if chunk else "",
        )

    return embeddings


def encode_text(text: str) -> list[float]:
    embeddings = encode_texts([text])
    if not embeddings:
        raise RuntimeError("SiliconFlow embedding 未返回向量。")
    return embeddings[0]


def ocr_pipeline(document_id: int, pdf_path: str) -> None:
    update_ocr_status(
        document_id,
        "processing",
        progress=1,
        detail="正在初始化索引任务。",
    )
    try:
        document = get_document(document_id)
        if not document:
            raise RuntimeError(f"文档不存在：{document_id}")

        delete_page_ocr(document_id)
        total_pages = int(document["page_count"])
        use_chunking = total_pages > PADDLEOCR_CHUNK_PAGE_THRESHOLD
        update_ocr_status(
            document_id,
            "processing",
            progress=3,
            detail=(
                f"文档共 {total_pages} 页，正在按分段模式提交 OCR 任务。"
                if use_chunking
                else "正在提交 OCR 任务。"
            ),
        )

        def handle_ocr_progress(
            *,
            state: str | None,
            extracted_pages: int,
            total_pages: int,
        ) -> None:
            effective_total = total_pages or int(document["page_count"])
            effective_done = min(max(extracted_pages, 0), effective_total) if effective_total else 0
            if state == "pending":
                update_ocr_status(
                    document_id,
                    "processing",
                    progress=5,
                    detail="OCR 任务排队中，等待服务开始处理。",
                )
                return
            if state == "running":
                progress = 5
                if effective_total > 0:
                    progress = 5 + int((effective_done / effective_total) * 65)
                update_ocr_status(
                    document_id,
                    "processing",
                    progress=progress,
                    detail=f"正在执行 索引：{effective_done}/{effective_total} 页。",
                )
                return
            if state == "done":
                update_ocr_status(
                    document_id,
                    "processing",
                    progress=70,
                    detail=f"OCR 已完成，正在解析 {effective_total} 页结果。",
                )

        def handle_ocr_retry(
            *,
            attempt: int,
            max_attempts: int,
            exc: Exception,
            delay: int,
            job_label: str,
        ) -> None:
            update_ocr_status(
                document_id,
                "processing",
                progress=5,
                detail=(
                    f"{job_label} 失败，正在重试"
                    f"（{attempt}/{max_attempts}，{delay} 秒后）: {exc}"
                ),
            )

        pages = collect_ocr_pages(
            pdf_path,
            total_pages=int(document["page_count"]),
            progress_callback=handle_ocr_progress,
            retry_callback=handle_ocr_retry,
        )
        total_pages = len(pages)
        logger.info(
            "[ocr.pipeline] document_id=%s parsed_pages=%s pdf=%s",
            document_id,
            total_pages,
            pdf_path,
        )

        non_empty_pages: list[tuple[int, str]] = []
        for index, page in enumerate(pages, start=1):
            page_number = int(page["page_number"])
            text = str(page["text"])
            save_page_ocr(document_id, page_number, text)
            if text.strip():
                non_empty_pages.append((page_number, text))

            progress = 70 + int((index / max(total_pages, 1)) * 10)
            update_ocr_status(
                document_id,
                "processing",
                progress=progress,
                detail=f"正在写入全文索引：{index}/{total_pages} 页。",
            )

        if non_empty_pages:
            processed_vectors = 0
            total_vectors = len(non_empty_pages)
            logger.info(
                "[ocr.pipeline] document_id=%s non_empty_pages=%s start_vector_index",
                document_id,
                total_vectors,
            )
            for start in range(0, total_vectors, SILICONFLOW_BATCH_SIZE):
                batch = non_empty_pages[start : start + SILICONFLOW_BATCH_SIZE]
                batch_embeddings = encode_texts([text for _, text in batch])
                for (page_number, _), embedding in zip(batch, batch_embeddings):
                    save_page_vector(document_id, page_number, embedding)
                processed_vectors += len(batch)
                progress = 80 + int((processed_vectors / total_vectors) * 19)
                update_ocr_status(
                    document_id,
                    "processing",
                    progress=progress,
                    detail=f"正在生成向量索引：{processed_vectors}/{total_vectors} 页。",
                )

        update_ocr_status(
            document_id,
            "processing",
            progress=99,
            detail="页级索引已完成，正在生成文档画像。",
        )
        try:
            update_document_profile(
                document_id,
                profile_status="processing",
                profile_detail="页级索引已完成，正在生成文档画像。",
            )
            build_document_profile(document_id)
        except Exception as profile_exc:  # noqa: BLE001
            logger.warning(
                "[profile.failed] document_id=%s error=%s",
                document_id,
                profile_exc,
            )
            update_document_profile(
                document_id,
                profile_status="failed",
                profile_detail=f"文档画像不可用：{profile_exc}",
            )
            update_ocr_status(
                document_id,
                "done",
                progress=100,
                detail=f"页级索引已完成，文档画像不可用：{profile_exc}",
            )
        else:
            try:
                update_ocr_status(
                    document_id,
                    "processing",
                    progress=99,
                    detail="文档画像已生成，正在生成内容分布。",
                )
                build_chapter_summary(document_id)
            except Exception as summary_exc:  # noqa: BLE001
                logger.warning(
                    "[chapter_summary.failed] document_id=%s error=%s",
                    document_id,
                    summary_exc,
                )
                update_ocr_status(
                    document_id,
                    "done",
                    progress=100,
                    detail=f"页级索引和文档画像已完成，内容分布生成失败：{summary_exc}",
                )
            else:
                update_ocr_status(
                    document_id,
                    "done",
                    progress=100,
                    detail="索引构建完成，可进行关键词、全文、向量检索和文档路由。",
                )
    except Exception as exc:  # noqa: BLE001
        update_ocr_status(
            document_id,
            "failed",
            progress=0,
            detail=f"索引构建失败：{exc}",
        )
        raise


def retrieve_pages(
    question: str,
    document_id: int,
    total_pages: int,
    fts_top_k: int = 4,
    vec_top_k: int = 4,
    window: int = 2,
    max_pages: int = 16,
    allow_fallback: bool = True,
    query_plan: dict[str, Any] | None = None,
) -> list[int]:
    page_text_map = get_page_ocr_text_map(document_id)
    if query_plan is None:
        query_plan = {}
        try:
            query_plan = understand_retrieval_query(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[query.understanding.error] question=%r error=%s",
                preview_text(question),
                exc,
            )

    plan_scope = str(query_plan.get("scope") or "").strip().lower()
    scope = plan_scope if plan_scope in {"targeted", "broad", "overview"} else detect_question_scope(question)
    query_variants = build_query_variants(question, query_plan=query_plan)
    if scope == "overview":
        fts_top_k = max(fts_top_k, 8)
        vec_top_k = max(vec_top_k, 8)
        max_pages = max(max_pages, 24)
    elif scope == "broad":
        fts_top_k = max(fts_top_k, 6)
        vec_top_k = max(vec_top_k, 6)
        max_pages = max(max_pages, 20)

    fts_pages: list[int] = []
    phrase_pages: list[int] = []
    seen_text_pages: set[int] = set()

    for query in query_variants or [question]:
        for page_number in fts_search_pages(query, document_id, top_k=fts_top_k):
            if page_number in seen_text_pages:
                continue
            seen_text_pages.add(page_number)
            fts_pages.append(page_number)
        for page_number in phrase_search_pages(query, document_id, top_k=max(2, fts_top_k)):
            if page_number in seen_text_pages:
                continue
            seen_text_pages.add(page_number)
            phrase_pages.append(page_number)

    keyword_pages = score_keyword_match_pages(
        question,
        query_plan=query_plan,
        page_text_map=page_text_map,
        top_k=max(fts_top_k * 2, 8),
    )

    vec_pages: list[int] = []
    variant_queries = [query.strip() for query in (query_variants or [question.strip()]) if query.strip()]
    if variant_queries:
        try:
            variant_embeddings = encode_texts(variant_queries)
        except Exception:
            variant_embeddings = []

        vec_scores: dict[int, int] = {}
        vec_first_seen: dict[int, tuple[int, int]] = {}
        for variant_index, embedding in enumerate(variant_embeddings):
            try:
                candidate_pages = vector_search_pages(
                    embedding,
                    document_id,
                    top_k=vec_top_k,
                )
            except Exception:
                continue

            variant_weight = max(1, len(variant_queries) - variant_index)
            for rank, page_number in enumerate(candidate_pages):
                rank_weight = max(1, vec_top_k - rank)
                vec_scores[page_number] = vec_scores.get(page_number, 0) + rank_weight + variant_weight
                vec_first_seen.setdefault(page_number, (variant_index, rank))

        vec_pages = sorted(
            vec_scores,
            key=lambda page_number: (
                -vec_scores[page_number],
                vec_first_seen[page_number][0],
                vec_first_seen[page_number][1],
                page_number,
            ),
        )

    seen: set[int] = set()
    hit_pages: list[int] = []
    for page_number in keyword_pages + phrase_pages + fts_pages + vec_pages:
        if page_number in seen:
            continue
        seen.add(page_number)
        hit_pages.append(page_number)

    if not hit_pages and scope != "overview":
        if not allow_fallback:
            logger.info(
                "[retrieval.no_hits] document_id=%s scope=%s question=%r fallback=disabled",
                document_id,
                scope,
                preview_text(question),
            )
            return []
        fallback_pages = collect_sparse_fallback_pages(
            document_id=document_id,
            total_pages=total_pages,
            max_pages=max(max_pages, 16),
        )
        logger.warning(
            "[retrieval.fallback] document_id=%s scope=%s question=%r fts_pages=%s vec_pages=%s fallback_pages=%s",
            document_id,
            scope,
            preview_text(question),
            phrase_pages + fts_pages,
            vec_pages,
            fallback_pages,
        )
        if fallback_pages:
            return fallback_pages
        return []

    overview_pages = (
        collect_overview_pages(
            document_id,
            total_pages,
            max_pages=max_pages if scope == "overview" else min(max_pages, 12),
        )
        if scope in {"overview", "broad"}
        else []
    )
    if not hit_pages and overview_pages:
        hit_pages = overview_pages[: min(len(overview_pages), 8)]

    secondary_queries, secondary_pages = expand_cross_reference_pages(
        document_id=document_id,
        seed_pages=hit_pages,
    )

    def expand_in_order(hits: list[int], width: int) -> list[int]:
        offsets = [0]
        for step in range(1, width + 1):
            offsets.extend([-step, step])

        ordered: list[int] = []
        seen_local: set[int] = set()
        for page_number in hits:
            for offset in offsets:
                candidate = page_number + offset
                if not (1 <= candidate <= total_pages):
                    continue
                if candidate in seen_local:
                    continue
                seen_local.add(candidate)
                ordered.append(candidate)
        return ordered

    target_like_seed_pages = [
        page_number
        for page_number in hit_pages
        if is_reference_target_page(page_text_map.get(page_number, ""))
    ]
    form_like_seed_pages = [
        page_number
        for page_number in hit_pages
        if is_form_template_page(page_text_map.get(page_number, ""))
    ]
    strong_keyword_seed_pages = keyword_pages[: min(len(keyword_pages), 3)]
    content_focused = is_content_focused_question(question, query_plan=query_plan)

    priority_groups = [
        overview_pages,
        expand_in_order(strong_keyword_seed_pages, 2 if content_focused else 1),
        expand_in_order(form_like_seed_pages, 1) if content_focused else [],
        expand_in_order(secondary_pages, 1),
        expand_in_order(target_like_seed_pages, 1),
        expand_in_order(hit_pages, 1 if secondary_pages else window),
    ]

    final_pages: list[int] = []
    seen_final: set[int] = set()
    for group in priority_groups:
        for page_number in group:
            if page_number in seen_final:
                continue
            if is_toc_like_page(page_text_map.get(page_number, "")):
                continue
            seen_final.add(page_number)
            final_pages.append(page_number)
            if len(final_pages) >= max_pages:
                break
        if len(final_pages) >= max_pages:
            break

    logger.info(
        "[retrieval] document_id=%s scope=%s question=%r keyword_pages=%s fts_pages=%s vec_pages=%s overview_pages=%s secondary_queries=%s secondary_pages=%s context_pages=%s",
        document_id,
        scope,
        preview_text(question),
        keyword_pages,
        phrase_pages + fts_pages,
        vec_pages,
        overview_pages,
        secondary_queries,
        secondary_pages,
        final_pages,
    )
    return final_pages


def build_chat_request(
    render_dir: str | Path,
    document: dict,
    question: str,
    conversation_history: list[dict[str, Any]],
    base_url: str = DEFAULT_BASE_URL,
    api_max_pixels_per_page: Optional[int] = LLM_RENDER_MAX_PIXELS,
    max_history_messages: int = 12,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[OpenAI, list[dict[str, Any]], list[dict[str, Any]]]:
    client = build_openai_client(base_url)

    rendered_pages = load_rendered_page_data_urls(
        render_dir,
        max_pixels=api_max_pixels_per_page,
        jpeg_quality=LLM_RENDER_JPEG_QUALITY,
    )
    if not rendered_pages:
        raise ValueError("当前文档没有可用的渲染图片，请重新上传 PDF。")
    page_images = make_page_image_items(document, rendered_pages)

    messages = build_chat_messages(
        page_images=page_images,
        question=question,
        conversation_history=conversation_history,
        api_max_pixels_per_page=api_max_pixels_per_page,
        max_history_messages=max_history_messages,
        system_prompt=system_prompt,
        document_summaries={
            int(document["id"]): document.get("chapter_summary") or get_chapter_summary(int(document["id"]))
        },
    )
    logger.info("[chat.request.full] question=%r page_count=%s history_messages=%s", preview_text(question), len(page_images), len(conversation_history))
    return client, messages, page_images


def build_chat_request_v2(
    render_dir: str | Path,
    document: dict,
    document_id: int,
    question: str,
    conversation_history: list[dict[str, Any]],
    total_pages: int,
    base_url: str = DEFAULT_BASE_URL,
    api_max_pixels_per_page: Optional[int] = LLM_RENDER_MAX_PIXELS,
    max_history_messages: int = 12,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[OpenAI, list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    context_pages = retrieve_pages(
        question=question,
        document_id=document_id,
        total_pages=total_pages,
    )

    if not context_pages:
        logger.warning(
            "[chat.request.full-fallback] document_id=%s question=%r reason=no_context_pages",
            document_id,
            preview_text(question),
        )
        client, messages, page_images = build_chat_request(
            render_dir=render_dir,
            document=document,
            question=question,
            conversation_history=conversation_history,
            base_url=base_url,
            api_max_pixels_per_page=api_max_pixels_per_page,
            max_history_messages=max_history_messages,
            system_prompt=system_prompt,
        )
        return client, messages, page_images, []

    client = build_openai_client(base_url)
    rendered_pages = load_selected_rendered_page_data_urls(
        render_dir=render_dir,
        page_numbers=context_pages,
        max_pixels=api_max_pixels_per_page,
        jpeg_quality=LLM_RENDER_JPEG_QUALITY,
    )
    if not rendered_pages:
        raise ValueError("索引命中的页面图片不存在，请重建索引后重试。")
    page_images = make_page_image_items(document, rendered_pages)

    messages = build_chat_messages(
        page_images=page_images,
        question=question,
        conversation_history=conversation_history,
        api_max_pixels_per_page=api_max_pixels_per_page,
        max_history_messages=max_history_messages,
        system_prompt=system_prompt,
        document_summaries={
            int(document["id"]): document.get("chapter_summary") or get_chapter_summary(int(document["id"]))
        },
    )
    logger.info(
        "[chat.request.retrieval] document_id=%s question=%r context_pages=%s page_images=%s history_messages=%s",
        document_id,
        preview_text(question),
        context_pages,
        len(page_images),
        len(conversation_history),
    )
    return client, messages, page_images, context_pages


def build_answer_sources(context_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for item in context_sources:
        document_id = int(item["document_id"])
        entry = grouped.setdefault(
            document_id,
            {
                "document_id": document_id,
                "document_name": item.get("document_name") or "",
                "document_display_name": item.get("document_display_name") or "",
                "pdf_url": item.get("pdf_url") or "",
                "pages": [],
                "score": item.get("score", 0),
                "rank": item.get("rank", 0),
            },
        )
        entry["score"] = max(int(entry.get("score", 0)), int(item.get("score", 0)))
        entry["rank"] = min(
            int(entry.get("rank", item.get("rank", 0) or 0) or 0),
            int(item.get("rank", 0) or 0),
        )
        page_number = int(item["page_number"])
        if page_number not in entry["pages"]:
            entry["pages"].append(page_number)

    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            int(item.get("rank", 999) or 999),
            -int(item.get("score", 0) or 0),
            item.get("document_display_name") or item.get("document_name") or "",
        ),
    )
    for item in ordered:
        item["pages"] = sorted(item["pages"])
    return ordered


def append_sources_block(answer: str, answer_sources: list[dict[str, Any]]) -> str:
    if not answer_sources:
        return answer

    lines = ["", "", "主要来源："]
    for source in answer_sources:
        name = source.get("document_display_name") or source.get("document_name") or "文档"
        pages = ", ".join(f"第 {page} 页" for page in source.get("pages") or [])
        lines.append(f"- {name}：{pages}")
    return answer.rstrip() + "\n" + "\n".join(lines)


def filter_documents_by_relevance(
    question: str,
    doc_pages: list[tuple[dict[str, Any], list[int]]],
    base_url: str = DEFAULT_BASE_URL,
) -> list[tuple[dict[str, Any], list[int]]]:
    if len(doc_pages) <= 1:
        return doc_pages

    doc_descriptions: list[str] = []
    for document, pages in doc_pages:
        doc_id = int(document["id"])
        name = document.get("display_name") or document.get("file_name") or ""
        summary = document.get("summary_text") or document.get("profile_detail") or ""
        chapter_summary = document.get("chapter_summary") or ""
        doc_type = document.get("doc_type") or ""
        keywords = "、".join(str(item) for item in (document.get("keywords") or []) if str(item).strip())
        aliases = "、".join(str(item) for item in (document.get("title_aliases") or []) if str(item).strip())
        page_list = ", ".join(str(page) for page in pages[:12])
        doc_descriptions.append(
            f"[document_id={doc_id}]\n"
            f"文件名：{name}\n"
            f"文档类型：{doc_type}\n"
            f"摘要：{summary}\n"
            f"关键词：{keywords}\n"
            f"标题别名：{aliases}\n"
            f"内容分布：\n{chapter_summary}\n"
            f"已召回页码：{page_list}"
        )

    started_at = time.perf_counter()
    try:
        user_message = f"用户问题：{question}\n\n" + "\n\n".join(doc_descriptions)
        client = build_openai_client(base_url)
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": DOCUMENT_RELEVANCE_FILTER_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=128,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        raw_content = extract_response_text(response.choices[0].message.content)
        parsed = _extract_json_object(raw_content)
        relevant_ids = {
            int(document_id)
            for document_id in (parsed.get("relevant_document_ids") or [])
            if str(document_id).strip().isdigit()
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[doc.relevance_filter.error] question=%r error=%s", preview_text(question), exc)
        return doc_pages

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    if not relevant_ids:
        logger.warning(
            "[doc.relevance_filter] elapsed_ms=%s question=%r all_filtered_out fallback=all",
            elapsed_ms,
            preview_text(question),
        )
        return doc_pages

    filtered = [(document, pages) for document, pages in doc_pages if int(document["id"]) in relevant_ids]
    logger.info(
        "[doc.relevance_filter] elapsed_ms=%s question=%r before=%s after=%s relevant_ids=%s",
        elapsed_ms,
        preview_text(question),
        [int(document["id"]) for document, _ in doc_pages],
        [int(document["id"]) for document, _ in filtered],
        sorted(relevant_ids),
    )
    return filtered if filtered else doc_pages


def retrieve_multi_document_context(
    question: str,
    routed_documents: list[dict[str, Any]],
    total_page_budget: int = MULTI_DOC_TOTAL_PAGE_BUDGET,
    per_doc_page_limit: int = MULTI_DOC_PER_DOC_PAGE_LIMIT,
    base_url: str = DEFAULT_BASE_URL,
) -> list[dict[str, Any]]:
    single_document = len(routed_documents) == 1
    shared_query_plan: dict[str, Any] = {}
    try:
        shared_query_plan = understand_retrieval_query(question)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[multi.query.understanding.error] question=%r error=%s", preview_text(question), exc)

    doc_pages: list[tuple[dict[str, Any], list[int]]] = []
    for document in routed_documents:
        pages = retrieve_pages(
            question=question,
            document_id=int(document["id"]),
            total_pages=int(document["page_count"]),
            allow_fallback=single_document,
            query_plan=shared_query_plan,
            max_pages=MULTI_DOC_SINGLE_DOC_PAGE_LIMIT,
        )
        if not pages:
            continue
        doc_pages.append((document, pages))

    if not single_document:
        doc_pages = filter_documents_by_relevance(question, doc_pages, base_url=base_url)
    doc_pages.sort(key=lambda item: -len(item[1]))

    effective_per_doc_limit = MULTI_DOC_SINGLE_DOC_PAGE_LIMIT if len(doc_pages) == 1 else per_doc_page_limit
    effective_total_budget = max(total_page_budget, effective_per_doc_limit)

    context_sources: list[dict[str, Any]] = []
    for rank_index, (document, pages) in enumerate(doc_pages, start=1):
        for local_rank, page_number in enumerate(pages[:effective_per_doc_limit], start=1):
            context_sources.append(
                {
                    "document_id": int(document["id"]),
                    "document_name": document.get("file_name") or "",
                    "document_display_name": document.get("display_name") or document.get("file_name") or "",
                    "page_number": int(page_number),
                    "rank": rank_index,
                    "score": max(1, effective_per_doc_limit - local_rank + 1),
                    "pdf_url": document.get("pdf_url") or "",
                    "render_dir": document.get("render_dir") or "",
                }
            )

    ordered = sorted(
        context_sources,
        key=lambda item: (
            int(item.get("rank", 999)),
            -int(item.get("score", 0)),
            int(item.get("page_number", 0)),
        ),
    )
    return ordered[:effective_total_budget]


def build_chat_request_multi(
    question: str,
    conversation_history: list[dict[str, Any]],
    base_url: str = DEFAULT_BASE_URL,
    api_max_pixels_per_page: Optional[int] = LLM_RENDER_MAX_PIXELS,
    max_history_messages: int = 12,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[OpenAI, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    routed_documents = route_documents(question)
    if not routed_documents:
        raise ValueError("当前没有可用于多文档路由的已完成索引文档。")

    context_sources = retrieve_multi_document_context(
        question,
        routed_documents,
        base_url=base_url,
    )
    if not context_sources:
        fallback_document = routed_documents[0]
        client, messages, page_images = build_chat_request(
            render_dir=fallback_document["render_dir"],
            document=fallback_document,
            question=question,
            conversation_history=conversation_history,
            base_url=base_url,
            api_max_pixels_per_page=api_max_pixels_per_page,
            max_history_messages=max_history_messages,
            system_prompt=system_prompt,
        )
        return client, messages, page_images, routed_documents, build_answer_sources([])

    page_images: list[dict[str, Any]] = []
    for source in context_sources:
        document = next(
            (item for item in routed_documents if int(item["id"]) == int(source["document_id"])),
            None,
        )
        if not document:
            continue
        rendered_pages = load_selected_rendered_page_data_urls(
            render_dir=document["render_dir"],
            page_numbers=[int(source["page_number"])],
            max_pixels=api_max_pixels_per_page,
            jpeg_quality=LLM_RENDER_JPEG_QUALITY,
        )
        for item in make_page_image_items(document, rendered_pages):
            page_images.append(item)

    if not page_images:
        raise ValueError("多文档命中的页面图片不存在，请重建索引后重试。")

    survived_doc_ids = {int(source["document_id"]) for source in context_sources}
    document_summaries: dict[int, str] = {}
    for document in routed_documents:
        doc_id = int(document["id"])
        if doc_id not in survived_doc_ids:
            continue
        summary = document.get("chapter_summary") or get_chapter_summary(doc_id)
        if summary:
            document_summaries[doc_id] = summary

    client = build_openai_client(base_url)
    messages = build_chat_messages(
        page_images=page_images,
        question=question,
        conversation_history=conversation_history,
        api_max_pixels_per_page=api_max_pixels_per_page,
        max_history_messages=max_history_messages,
        system_prompt=system_prompt,
        document_summaries=document_summaries,
    )
    return client, messages, page_images, routed_documents, build_answer_sources(context_sources)


def ask_qwen_about_rendered_pdf(
    render_dir: str | Path,
    document: dict,
    question: str,
    conversation_history: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_max_pixels_per_page: Optional[int] = LLM_RENDER_MAX_PIXELS,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    enable_thinking: bool = False,
    max_history_messages: int = 12,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    client, messages, page_images = build_chat_request(
        render_dir=render_dir,
        document=document,
        question=question,
        conversation_history=conversation_history,
        base_url=base_url,
        api_max_pixels_per_page=api_max_pixels_per_page,
        max_history_messages=max_history_messages,
        system_prompt=system_prompt,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"enable_thinking": enable_thinking},
    )

    answer = extract_response_text(response.choices[0].message.content)
    usage = usage_to_dict(response.usage)
    summary = build_summary(
        model=model,
        page_images=page_images,
        usage=usage,
        max_tokens=max_tokens,
    )

    return {
        "answer": answer,
        "usage": usage,
        "summary": summary,
        "summary_json": json.dumps(summary, ensure_ascii=False, indent=2),
    }
