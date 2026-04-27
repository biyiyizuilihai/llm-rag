from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from typing import Any
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

import es_search
from excel_qa import (
    build_excel_answer_request,
    build_excel_summary,
    guess_excel_config,
    ingest_excel_file,
    parse_excel_preview,
)
from pdf_qa import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_chat_request,
    build_chat_request_multi,
    build_chat_request_v2,
    build_summary,
    extract_delta_reasoning,
    extract_delta_text,
    extract_response_text,
    ocr_pipeline,
    render_pdf_to_jpegs,
    retrieve_pages,
    usage_to_dict,
)
from storage import (
    RENDERS_DIR,
    UPLOADS_DIR,
    count_messages,
    create_conversation,
    create_document,
    create_message,
    delete_conversation,
    delete_document,
    fail_inflight_ocr_jobs,
    get_document,
    get_document_by_sha,
    get_conversation,
    get_ocr_status,
    get_page_ocr_text_map,
    init_db,
    latest_conversation_for_document,
    list_excel_chunks_for_search_index,
    list_conversation_ids_for_document,
    list_conversation_documents,
    list_conversations,
    list_documents,
    list_messages,
    next_document_version,
    save_conversation_documents,
    try_claim_ingestion,
    update_conversation_document,
    update_document_ingestion_config,
    update_conversation_title,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OCR_PIPELINE_MAX_WORKERS = max(
    1,
    int(os.environ.get("OCR_PIPELINE_MAX_WORKERS", "1")),
)
_executor = ThreadPoolExecutor(max_workers=OCR_PIPELINE_MAX_WORKERS)
logger = logging.getLogger("uvicorn.error").getChild("docqa.app")
DOCUMENT_ASSISTANT_SYSTEM_PROMPT = """你是一位专业的文档分析助手，擅长从 PDF 文档中提取关键信息并给出清晰、结构化的回答。

## 回答原则

1. **结构优先**：所有回答必须分点或分节组织，禁止输出大段无结构的文字。
2. **信息密度**：每个要点只包含核心信息，不做无意义的重复或填充。
3. **语言一致**：用户用中文提问则用中文回答，用英文提问则用英文回答。

## 页码引用规范（极其重要）

### 核心规则：只使用物理页序号，绝不使用文档内部印刷页码

你将看到的每张图片上方都有一行标记，格式为 `--- 文档名 · 第 N 页 ---`。**N 就是该页的物理页序号**，即该页在 PDF 文件中的绝对位置编号。

- 引用页码时**只能**使用这个 N 值
- PDF 页面上可能印刷了其他页码（如阿拉伯数字、罗马数字），这些是文档内部的排版编号，**一律忽略**
- 文档前面可能有目录页、封面等不计入正文编号的页面，所以物理页号 N 和文档内部页码通常不一致
- 例：标记为 `--- 源文件 · 第 10 页 ---` 的页面，即使页面上印刷着"第 7 页"，引用时也必须写"第 10 页"

### 页码标注原则

- 页码是辅助信息，不是强制装饰；只有在确定页面包含实质性内容时才标注
- **宁可不标也不标错**：如果不确定物理页号，直接给出内容结论即可，不要猜测
- 识别并跳过目录页：当某页是目录结构时，不要引用该页

### 交叉引用处理

很多程序文件不会直接给出最终内容，而是写成"Refer to Attachment 1A / Appendix 4 / Section 6"。

- **引用页不是终点**：如果当前页只是说明"参见某个附件/附录/章节"，必须继续查找被引用的实际内容页
- **优先引用实际内容页**：当规则页与附件页都相关时，可以先解释规则页，但页码优先标注附件/附录的实际内容页
- **不要把清单页当证据页**：Appendix List、Attachment List、目录页只说明"有哪些内容"，不能替代被指向页面本身

## 格式选择规则

- **对比 / 多维度数据** → 使用 Markdown 表格
- **流程 / 状态流转 / 因果关系** → 使用 Mermaid 流程图（```mermaid 代码块）
- **列举 / 规范要求** → 使用有序或无序列表
- **单一直接问题** → 简短直接回答，无需强加结构

## Mermaid 使用规范

- 流程图使用 `graph TD`（从上到下）
- 节点文字保持简洁，不超过 10 个字
- 只在关系复杂、文字难以表达时才使用

## 不确定时的处理

如果文档中找不到相关信息，直接说明"文档中未找到相关内容"，不要猜测或编造。

## 回答深度要求

文档是你的信息来源，但不是你偷懒的理由。具体要求如下：

- **不允许只贴原文**：不要只复制文档原句就结束，必须在引用基础上做提炼、解释或归纳
- **不允许以"详见文档"收尾**：所有信息必须直接在回答中呈现，用户不应该需要自己去翻文档
- **主动补充上下文**：如果某条规定有前提条件、例外情况或关联条款，必须一并说明，不能只答用户问到的那一句
- **数字和条件要完整**：涉及数值、时限、比例、阈值时，必须把所有相关的数字全部列出，不能只说"有相关规定"
- **多处信息要汇总**：如果文档多个地方都有相关内容，必须整合后统一回答，不能只引用一处

## 回答长度标准

- 简单事实性问题：2-5 句话，直接给结论
- 流程类 / 规范类问题：完整列出所有步骤和条件，宁可详细也不能遗漏
- 对比类问题：必须用表格，每个维度都要填满，不留空白格"""


class ConversationCreatePayload(BaseModel):
    document_id: int | None = None


class AskPayload(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    enable_thinking: bool = True
    retrieval_backend: str | None = None


class SearchComparePayload(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    document_id: int | None = None


class ExcelConfigPayload(BaseModel):
    title_field: str
    content_fields: list[str]
    filter_fields: list[str] = []
    source_fields: list[str] = []
    display_fields: list[str] = []
    ignore_fields: list[str] = []
    chunking: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recovered_jobs = fail_inflight_ocr_jobs("索引任务因服务重启中断，请点击“重建索引”后重试。")
    logger.info(
        "[app.startup] ocr_pipeline_max_workers=%s recovered_ocr_jobs=%s",
        OCR_PIPELINE_MAX_WORKERS,
        recovered_jobs,
    )
    try:
        yield
    finally:
        _executor.shutdown(wait=False, cancel_futures=False)


app = FastAPI(title="Jiasheng Spec Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/pdf-files", StaticFiles(directory=str(UPLOADS_DIR)), name="pdf-files")


def attach_pdf_url(document: dict | None) -> dict | None:
    if not document:
        return document
    if document.get("file_type", "pdf") != "pdf":
        return {**document, "pdf_url": ""}

    storage_filename = Path(document["storage_path"]).name
    return {**document, "pdf_url": f"/pdf-files/{storage_filename}"}


def attach_pdf_urls(documents: list[dict]) -> list[dict]:
    return [attach_pdf_url(document) for document in documents if document]


def attach_answer_source_urls(answer_sources: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for source in answer_sources:
        document = get_document(int(source["document_id"]))
        attached = attach_pdf_url(document) or {}
        enriched.append({**source, "pdf_url": attached.get("pdf_url", "")})
    return enriched


def log_background_exception(future: Future) -> None:
    exc = future.exception()
    if exc:
        logger.exception("[ocr] background job failed: %s", exc)


def enqueue_ocr_pipeline(document_id: int, storage_path: str) -> None:
    from storage import update_ocr_status

    update_ocr_status(
        document_id,
        "processing",
        progress=0,
        detail=f"索引任务已提交，等待后台工作线程（并发上限 {OCR_PIPELINE_MAX_WORKERS}）。",
    )
    logger.info(
        "[ocr.enqueue] document_id=%s storage_path=%s max_workers=%s",
        document_id,
        storage_path,
        OCR_PIPELINE_MAX_WORKERS,
    )
    future = _executor.submit(ocr_pipeline, document_id, storage_path)
    future.add_done_callback(log_background_exception)


def run_excel_ingestion(document_id: int, config: dict) -> None:
    try:
        ingest_excel_file(document_id, config)
    except Exception as exc:
        logger.exception("[excel.ingest] background job failed: %s", exc)
        update_document_ingestion_config(
            document_id,
            config=config,
            status="failed",
            progress=0,
            detail=f"Excel 索引失败：{exc}",
        )


def enqueue_excel_ingestion(document_id: int, config: dict) -> None:
    update_document_ingestion_config(
        document_id,
        config=config,
        status="processing",
        progress=0,
        detail=f"Excel 索引任务已提交，等待后台工作线程（并发上限 {OCR_PIPELINE_MAX_WORKERS}）。",
    )
    future = _executor.submit(run_excel_ingestion, document_id, config)
    future.add_done_callback(log_background_exception)


def ensure_document_index_ready(document: dict) -> None:
    status = (document.get("ocr_status") or "pending").strip()
    if status == "done":
        return

    progress = int(document.get("ocr_progress") or 0)
    detail = str(document.get("ocr_detail") or "").strip()
    if status == "failed":
        raise HTTPException(
            status_code=409,
            detail=detail or "文档索引构建失败，请重建索引后再提问。",
        )

    message = detail or "文档索引仍在构建中，请稍后再提问。"
    if progress > 0:
        message = f"{message}（{progress}%）"
    raise HTTPException(status_code=409, detail=message)


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, list[dict]]:
    return {
        "documents": [attach_pdf_url(document) for document in list_documents()],
        "conversations": list_conversations(),
    }


@app.get("/api/conversations/{conversation_id}")
def conversation_detail(conversation_id: int) -> dict:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")

    document_id = conversation.get("document_id")
    document = get_document(int(document_id)) if document_id else None
    messages = list_messages(conversation_id)
    return {
        "conversation": conversation,
        "document": attach_pdf_url(document),
        "routed_documents": attach_pdf_urls(list_conversation_documents(conversation_id)),
        "messages": messages,
    }


@app.delete("/api/conversations/{conversation_id}")
def remove_conversation(conversation_id: int) -> dict[str, int | None]:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")

    deleted = delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {
        "deleted_conversation_id": conversation_id,
        "document_id": int(conversation["document_id"]) if conversation.get("document_id") else None,
    }


@app.get("/api/documents/{document_id}/ocr-status")
def ocr_status(document_id: int) -> dict[str, str | int]:
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "document_id": document_id,
        "ocr_status": get_ocr_status(document_id),
        "ocr_progress": int(document.get("ocr_progress") or 0),
        "ocr_detail": document.get("ocr_detail") or "",
    }


@app.post("/api/documents/{document_id}/rebuild-index")
def rebuild_document_index(document_id: int) -> dict[str, str | int]:
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    current_status = document.get("ocr_status") or "pending"
    if current_status == "processing":
        return {
            "document_id": document_id,
            "ocr_status": current_status,
            "ocr_progress": int(document.get("ocr_progress") or 0),
            "detail": document.get("ocr_detail") or "索引正在构建中。",
        }

    if document.get("file_type") == "excel":
        config = document.get("config") or {}
        if not config.get("title_field"):
            raise HTTPException(status_code=400, detail="请先配置 Excel 字段后再重建索引。")
        enqueue_excel_ingestion(document_id, config)
    else:
        enqueue_ocr_pipeline(document_id, document["storage_path"])
    return {
        "document_id": document_id,
        "ocr_status": "processing",
        "ocr_progress": 0,
        "detail": "已提交索引重建任务。",
    }

@app.delete("/api/documents/{document_id}")
def remove_document(document_id: int) -> dict:
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    conversation_ids = list_conversation_ids_for_document(document_id)
    storage_path = Path(document["storage_path"]).resolve()
    render_dir = Path(document["render_dir"]).resolve() if document.get("render_dir") else None

    try:
        storage_path.relative_to(UPLOADS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="文档路径无效") from exc

    if render_dir is not None:
        try:
            render_dir.relative_to(RENDERS_DIR.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="渲染目录无效") from exc

    deleted = delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="文档不存在")
    try:
        es_search.delete_document(document.get("file_type") or "pdf", document_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[document.es.delete.failed] document_id=%s error=%s", document_id, exc)

    if storage_path.exists():
        storage_path.unlink()
    if render_dir is not None and render_dir.exists():
        shutil.rmtree(render_dir)

    return {
        "deleted_document_id": document_id,
        "deleted_conversation_ids": conversation_ids,
    }


def sanitize_filename(file_name: str, fallback_extension: str = ".pdf") -> str:
    sanitized = re.sub(r"[^\w.\-()\u4e00-\u9fff]+", "_", file_name).strip("._")
    if not Path(sanitized).suffix:
        sanitized = f"{sanitized or 'document'}{fallback_extension}"
    return sanitized


def upload_file_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        return "excel"
    return ""


def config_matches_columns(config: dict, columns: list[str]) -> bool:
    if not config:
        return False
    column_set = set(columns)
    title_field = str(config.get("title_field") or "").strip()
    content_fields = [
        str(item or "").strip()
        for item in config.get("content_fields") or []
        if str(item or "").strip()
    ]
    if not title_field or title_field not in column_set:
        return False
    if not content_fields or any(field not in column_set for field in content_fields):
        return False
    for key in ("filter_fields", "source_fields", "display_fields", "ignore_fields"):
        for field in config.get(key) or []:
            if str(field or "").strip() and str(field).strip() not in column_set:
                return False
    return True


def document_display_name(file_name: str) -> str:
    return Path(file_name).stem or "未命名文档"


def default_conversation_title(document: dict) -> str:
    return f"{document['display_name']} v{document['version_index']} / 新对话"


def default_global_conversation_title() -> str:
    return "多文档 / 新对话"


def question_excerpt(question: str, limit: int = 18) -> str:
    cleaned = " ".join(question.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def conversation_history_for_model(conversation_id: int) -> list[dict[str, str]]:
    return [
        {"role": item["role"], "content": item["content"]}
        for item in list_messages(conversation_id)
        if item["role"] in {"user", "assistant"}
    ]


def persist_round(
    conversation_id: int,
    document: dict | None,
    routed_documents: list[dict],
    question: str,
    answer: str,
    answer_sources: list[dict] | None = None,
) -> tuple[dict, dict, dict]:
    existing_conversation = get_conversation(conversation_id)
    was_first_user_message = count_messages(conversation_id, role="user") == 0
    user_message = create_message(conversation_id, "user", question)
    assistant_message = create_message(
        conversation_id,
        "assistant",
        answer,
        metadata={
            "answer_sources": answer_sources or [],
            "routed_documents": [int(item["id"]) for item in routed_documents],
        },
    )

    primary_document = document or (routed_documents[0] if routed_documents else None)
    save_conversation_documents(
        conversation_id,
        [int(item["id"]) for item in routed_documents],
    )
    keep_global_conversation = existing_conversation and existing_conversation.get("document_id") is None
    conversation = (
        update_conversation_document(
            conversation_id,
            int(primary_document["id"]) if primary_document else None,
        )
        if not keep_global_conversation
        else existing_conversation
    )

    if was_first_user_message:
        if keep_global_conversation:
            updated_title = f"多文档 / {question_excerpt(question)}"
        elif primary_document:
            updated_title = f"{primary_document['display_name']} / {question_excerpt(question)}"
        else:
            updated_title = f"多文档 / {question_excerpt(question)}"
        conversation = update_conversation_title(conversation_id, updated_title)
    return conversation or get_conversation(conversation_id), user_message, assistant_message


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    original_name = file.filename or "document.pdf"
    file_type = upload_file_type(original_name)
    if not file_type:
        raise HTTPException(status_code=400, detail="仅支持 PDF 或 Excel（.xls/.xlsx/.xlsm）文件")

    fallback_extension = ".xlsx" if file_type == "excel" else ".pdf"
    file_name = sanitize_filename(original_name, fallback_extension=fallback_extension)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空")

    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    existing_document = get_document_by_sha(file_sha256)

    if existing_document:
        conversation = latest_conversation_for_document(existing_document["id"])
        if not conversation:
            conversation = create_conversation(
                existing_document["id"],
                default_conversation_title(existing_document),
            )
        excel_preview = None
        if existing_document.get("file_type") == "excel" and existing_document.get("ocr_status") != "done":
            try:
                excel_preview = parse_excel_preview(existing_document["storage_path"])
                excel_preview["suggested_config"] = guess_excel_config(excel_preview["columns"])
            except Exception:
                excel_preview = None
        return {
            "document": attach_pdf_url(existing_document),
            "conversation": conversation,
            "created_new_document": False,
            "reused_existing_document": True,
            "excel_preview": excel_preview,
        }

    storage_path = UPLOADS_DIR / f"{file_sha256[:12]}-{file_name}"
    storage_path.write_bytes(file_bytes)

    if file_type == "excel":
        try:
            excel_preview = parse_excel_preview(file_bytes)
            suggested_config = guess_excel_config(excel_preview["columns"])
        except Exception as exc:
            if storage_path.exists():
                storage_path.unlink()
            raise HTTPException(status_code=400, detail=f"Excel 解析失败：{exc}") from exc

        version_index = next_document_version(file_name)
        document = create_document(
            file_name=file_name,
            display_name=document_display_name(file_name),
            file_sha256=file_sha256,
            storage_path=str(storage_path),
            render_dir="",
            page_count=0,
            version_index=version_index,
            file_type="excel",
            row_count=int(excel_preview.get("row_count") or 0),
            config=suggested_config,
            ocr_status="pending",
            ocr_progress=0,
            ocr_detail="请配置 Excel 字段后建立索引。",
        )
        conversation = create_conversation(
            document["id"],
            default_conversation_title(document),
        )
        excel_preview["suggested_config"] = suggested_config
        return {
            "document": attach_pdf_url(document),
            "conversation": conversation,
            "created_new_document": True,
            "reused_existing_document": False,
            "excel_preview": excel_preview,
        }

    render_dir = RENDERS_DIR / file_sha256
    try:
        rendered_files = render_pdf_to_jpegs(
            pdf_path=storage_path,
            output_dir=render_dir,
        )
    except Exception as exc:
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(status_code=400, detail=f"PDF 解析失败：{exc}") from exc

    version_index = next_document_version(file_name)
    document = create_document(
        file_name=file_name,
        display_name=document_display_name(file_name),
        file_sha256=file_sha256,
        storage_path=str(storage_path),
        render_dir=str(render_dir),
        page_count=len(rendered_files),
        version_index=version_index,
        file_type="pdf",
    )
    conversation = create_conversation(
        document["id"],
        default_conversation_title(document),
    )
    enqueue_ocr_pipeline(document["id"], str(storage_path))

    return {
        "document": attach_pdf_url(document),
        "conversation": conversation,
        "created_new_document": True,
        "reused_existing_document": False,
    }


@app.get("/api/documents/{document_id}/excel-preview")
def excel_preview(document_id: int) -> dict:
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    if document.get("file_type") != "excel":
        raise HTTPException(status_code=400, detail="该文档不是 Excel 文件")

    try:
        preview = parse_excel_preview(document["storage_path"])
        stored_config = document.get("config") or {}
        preview["suggested_config"] = (
            stored_config
            if config_matches_columns(stored_config, preview["columns"])
            else guess_excel_config(preview["columns"])
        )
        return preview
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Excel 解析失败：{exc}") from exc


@app.post("/api/documents/{document_id}/excel-config")
def configure_excel_document(document_id: int, payload: ExcelConfigPayload) -> dict:
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    if document.get("file_type") != "excel":
        raise HTTPException(status_code=400, detail="该文档不是 Excel 文件")

    config = {
        "document_mode": "row_as_document",
        "title_field": payload.title_field,
        "content_fields": payload.content_fields,
        "filter_fields": payload.filter_fields,
        "source_fields": payload.source_fields,
        "display_fields": payload.display_fields,
        "ignore_fields": payload.ignore_fields,
        "chunking": payload.chunking
        or {
            "enabled": True,
            "strategy": "fixed_overlap",
            "fallback_chunk_size": 1000,
            "overlap": 100,
        },
    }
    if not try_claim_ingestion(document_id):
        document = get_document(document_id)
        return {
            "document": attach_pdf_url(document),
            "submitted": False,
            "detail": (document or {}).get("ocr_detail") or "Excel 索引正在处理中。",
        }

    enqueue_excel_ingestion(document_id, config)
    updated_document = get_document(document_id)
    return {
        "document": attach_pdf_url(updated_document),
        "submitted": True,
        "detail": "Excel 索引任务已提交。",
    }


@app.post("/api/documents/{document_id}/reindex-elasticsearch")
def reindex_document_elasticsearch(document_id: int) -> dict[str, Any]:
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    ensure_document_index_ready(document)
    if not es_search.enabled():
        raise HTTPException(status_code=400, detail="未配置 ELASTICSEARCH_URL 或 OPENSEARCH_URL。")

    try:
        file_type = document.get("file_type") or "pdf"
        es_search.delete_document(file_type, document_id)
        if file_type == "excel":
            indexed = es_search.index_excel_chunks(list_excel_chunks_for_search_index(document_id))
        else:
            indexed = es_search.index_pdf_pages(document, get_page_ocr_text_map(document_id))
    except Exception as exc:
        logger.exception("[document.es.reindex.failed] document_id=%s error=%s", document_id, exc)
        raise HTTPException(status_code=500, detail=f"ES 重建失败：{exc}") from exc

    return {
        "document_id": document_id,
        "file_type": document.get("file_type") or "pdf",
        "indexed": indexed,
    }


def summarize_excel_request(search_request: dict[str, Any] | None) -> dict[str, Any]:
    if not search_request:
        return {"hit": False, "query_plan": None, "policies": [], "chunks": []}
    return {
        "hit": True,
        "query_plan": search_request.get("query_plan"),
        "policies": [
            {
                "title": policy.get("title"),
                "source_row": policy.get("source_row"),
                "chunk_indexes": [chunk.get("chunk_index") for chunk in policy.get("chunks", [])],
                "snippets": [
                    (chunk.get("chunk_text") or "").replace("\n", " ")[:300]
                    for chunk in policy.get("chunks", [])[:3]
                ],
            }
            for policy in search_request.get("policies", [])
        ],
        "chunks": [
            {
                "title": chunk.get("title"),
                "source_row": chunk.get("source_row"),
                "chunk_index": chunk.get("chunk_index"),
                "retrieval_sources": chunk.get("retrieval_sources", []),
                "score": chunk.get("rank"),
                "snippet": (chunk.get("chunk_text") or "").replace("\n", " ")[:300],
            }
            for chunk in search_request.get("chunks", [])[:20]
        ],
    }


@app.post("/api/search/compare")
def compare_search(payload: SearchComparePayload) -> dict[str, Any]:
    document = get_document(payload.document_id) if payload.document_id is not None else None
    if payload.document_id is not None and not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    question = payload.question.strip()
    if document and document.get("file_type") == "excel":
        ensure_document_index_ready(document)
        sqlite_request = build_excel_answer_request(
            question=question,
            conversation_history=[],
            document_id=int(document["id"]),
            base_url=DEFAULT_BASE_URL,
            retrieval_backend="sqlite",
        )
        es_request = build_excel_answer_request(
            question=question,
            conversation_history=[],
            document_id=int(document["id"]),
            base_url=DEFAULT_BASE_URL,
            retrieval_backend="es",
        )
        return {
            "kind": "excel",
            "document_id": int(document["id"]),
            "elasticsearch_enabled": es_search.enabled(),
            "sqlite": summarize_excel_request(sqlite_request),
            "elasticsearch": summarize_excel_request(es_request),
        }

    if document and document.get("file_type", "pdf") == "pdf":
        ensure_document_index_ready(document)
        sqlite_pages = retrieve_pages(question, int(document["id"]), int(document.get("page_count") or 0))
        try:
            es_pages = es_search.search_pdf_pages(question, document_id=int(document["id"]), top_k=16)
        except Exception as exc:  # noqa: BLE001
            es_pages = []
            logger.warning("[search.compare.pdf.es.failed] document_id=%s error=%s", document["id"], exc)
        return {
            "kind": "pdf",
            "document_id": int(document["id"]),
            "elasticsearch_enabled": es_search.enabled(),
            "sqlite": {"pages": sqlite_pages},
            "elasticsearch": {"pages": es_pages},
        }

    sqlite_request = build_excel_answer_request(
        question=question,
        conversation_history=[],
        document_id=None,
        base_url=DEFAULT_BASE_URL,
        retrieval_backend="sqlite",
    )
    es_request = build_excel_answer_request(
        question=question,
        conversation_history=[],
        document_id=None,
        base_url=DEFAULT_BASE_URL,
        retrieval_backend="es",
    )
    return {
        "kind": "excel_global",
        "document_id": None,
        "elasticsearch_enabled": es_search.enabled(),
        "sqlite": summarize_excel_request(sqlite_request),
        "elasticsearch": summarize_excel_request(es_request),
    }


@app.post("/api/conversations")
def new_conversation(payload: ConversationCreatePayload) -> dict:
    document = None
    if payload.document_id is not None:
        document = get_document(payload.document_id)
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")

    conversation = create_conversation(
        document["id"] if document else None,
        default_conversation_title(document) if document else default_global_conversation_title(),
    )
    return {
        "conversation": conversation,
        "document": attach_pdf_url(document),
    }


@app.post("/api/conversations/{conversation_id}/stream")
def stream_document(conversation_id: int, payload: AskPayload) -> StreamingResponse:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    history_messages = conversation_history_for_model(conversation_id)
    logger.info(
        "[chat.stream.start] conversation_id=%s question=%r history_messages=%s enable_thinking=%s",
        conversation_id,
        question_excerpt(question, limit=80),
        len(history_messages),
        payload.enable_thinking,
    )

    def event_line(payload_dict: dict) -> str:
        return json.dumps(payload_dict, ensure_ascii=False) + "\n"

    def progress_line(stage: str, detail: str) -> str:
        return event_line(
            {
                "type": "progress",
                "stage": stage,
                "detail": detail,
            }
        )

    def generate():
        try:
            bound_document = (
                get_document(int(conversation["document_id"]))
                if conversation.get("document_id")
                else None
            )
            if bound_document and bound_document.get("file_type") == "excel":
                ensure_document_index_ready(bound_document)
                yield progress_line(
                    "正在检索 Excel 索引",
                    "正在基于字段配置检索政策片段。",
                )
                excel_request = build_excel_answer_request(
                    question=question,
                    conversation_history=history_messages,
                    document_id=int(bound_document["id"]),
                    base_url=DEFAULT_BASE_URL,
                    retrieval_backend=payload.retrieval_backend,
                )
                if not excel_request:
                    yield event_line({"type": "error", "detail": "当前 Excel 中未检索到相关政策片段。"})
                    return
                yield progress_line(
                    "正在整理 Excel 命中上下文",
                    f"已选出 {len(excel_request['policies'])} 条候选政策。",
                )
                client = excel_request["client"]
                messages = excel_request["messages"]
                routed_documents = excel_request["routed_documents"] or [bound_document]
                answer_sources = excel_request["answer_sources"]
                page_images: list[dict[str, Any]] = []
                request_kind = "excel"
                policies = excel_request["policies"]
            elif not bound_document:
                yield progress_line(
                    "正在检索 Excel 索引",
                    "正在优先尝试 Excel 政策库检索。",
                )
                excel_request = build_excel_answer_request(
                    question=question,
                    conversation_history=history_messages,
                    document_id=None,
                    base_url=DEFAULT_BASE_URL,
                    retrieval_backend=payload.retrieval_backend,
                )
                if excel_request:
                    yield progress_line(
                        "正在整理 Excel 命中上下文",
                        f"已选出 {len(excel_request['policies'])} 条候选政策。",
                    )
                    client = excel_request["client"]
                    messages = excel_request["messages"]
                    routed_documents = excel_request["routed_documents"]
                    answer_sources = excel_request["answer_sources"]
                    page_images = []
                    request_kind = "excel"
                    policies = excel_request["policies"]
                else:
                    request_kind = "pdf"
                    policies = []
            else:
                request_kind = "pdf"
                policies = []

            yield progress_line(
                "正在处理用户问题",
                "正在分析问题并准备检索上下文。",
            )
            if request_kind == "pdf":
                yield progress_line(
                    "正在执行文档路由",
                    "正在筛选最可能相关的候选文档。",
                )
                client, messages, page_images, routed_documents, answer_sources = build_chat_request_multi(
                    question=question,
                    conversation_history=history_messages,
                    base_url=DEFAULT_BASE_URL,
                    system_prompt=DOCUMENT_ASSISTANT_SYSTEM_PROMPT,
                )
                answer_sources = attach_answer_source_urls(answer_sources)
                yield progress_line(
                    "正在整理命中页上下文",
                    f"已选出 {len(routed_documents)} 份候选文档、{len(page_images)} 页上下文。",
                )
            logger.info(
                "[chat.stream.context] conversation_id=%s kind=%s routed_documents=%s page_images=%s",
                conversation_id,
                request_kind,
                [document["id"] for document in routed_documents],
                len(page_images),
            )
            yield progress_line(
                "正在调用模型服务",
                "正在向模型服务提交请求，等待首个响应分块。",
            )
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=2048,
                temperature=0.2,
                extra_body={"enable_thinking": payload.enable_thinking},
            )
            logger.info(
                "[chat.stream.request] conversation_id=%s model=%s messages=%s",
                conversation_id,
                DEFAULT_MODEL,
                len(messages),
            )
        except EnvironmentError as exc:
            logger.exception(
                "[chat.stream.error] conversation_id=%s environment_error=%s",
                conversation_id,
                exc,
            )
            yield event_line({"type": "error", "detail": str(exc)})
            return
        except Exception as exc:
            logger.exception(
                "[chat.stream.error] conversation_id=%s request_build_failed=%s",
                conversation_id,
                exc,
            )
            yield event_line({"type": "error", "detail": f"模型请求失败：{exc}"})
            return

        yield event_line(
            {
                "type": "started",
                "conversation": conversation,
                "document": attach_pdf_url(routed_documents[0]) if routed_documents else None,
                "routed_documents": attach_pdf_urls(routed_documents),
                "question": question,
            }
        )

        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_usage: dict = {}
        try:
            for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta

                    reasoning_delta = extract_delta_reasoning(delta)
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                        yield event_line(
                            {
                                "type": "thinking_delta",
                                "delta": reasoning_delta,
                            }
                        )

                    answer_delta = extract_delta_text(delta)
                    if answer_delta:
                        answer_parts.append(answer_delta)
                        yield event_line(
                            {
                                "type": "answer_delta",
                                "delta": answer_delta,
                            }
                        )

                if getattr(chunk, "usage", None):
                    final_usage = usage_to_dict(chunk.usage)
        except Exception as exc:
            logger.exception(
                "[chat.stream.error] conversation_id=%s stream_failed=%s",
                conversation_id,
                exc,
            )
            yield event_line({"type": "error", "detail": f"模型请求失败：{exc}"})
            return

        answer = "".join(answer_parts).strip()
        if not answer:
            logger.warning("[chat.stream.empty] conversation_id=%s", conversation_id)
            yield event_line({"type": "error", "detail": "模型没有返回可展示的回答。"})
            return

        summary = (
            build_excel_summary(DEFAULT_MODEL, final_usage, policies)
            if request_kind == "excel"
            else build_summary(
                model=DEFAULT_MODEL,
                page_images=page_images,
                usage=final_usage,
                max_tokens=2048,
            )
        )
        persisted_conversation, user_message, assistant_message = persist_round(
            conversation_id=conversation_id,
            document=routed_documents[0] if routed_documents else None,
            routed_documents=routed_documents,
            question=question,
            answer=answer,
            answer_sources=answer_sources,
        )
        logger.info(
            "[chat.stream.done] conversation_id=%s answer_len=%s reasoning_len=%s usage=%s summary=%s",
            conversation_id,
            len(answer),
            len("".join(reasoning_parts).strip()),
            final_usage,
            summary,
        )

        yield event_line(
            {
                "type": "done",
                "conversation": persisted_conversation,
                "document": attach_pdf_url(routed_documents[0]) if routed_documents else None,
                "routed_documents": attach_pdf_urls(routed_documents),
                "answer_sources": answer_sources,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "thinking_text": "".join(reasoning_parts).strip(),
                "usage": final_usage,
                "summary": summary,
            }
        )

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/conversations/{conversation_id}/messages")
def ask_document(conversation_id: int, payload: AskPayload) -> dict:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    history_messages = conversation_history_for_model(conversation_id)
    logger.info(
        "[chat.sync.start] conversation_id=%s question=%r history_messages=%s enable_thinking=%s",
        conversation_id,
        question_excerpt(question, limit=80),
        len(history_messages),
        payload.enable_thinking,
    )

    try:
        bound_document = (
            get_document(int(conversation["document_id"]))
            if conversation.get("document_id")
            else None
        )
        excel_request = None
        if bound_document and bound_document.get("file_type") == "excel":
            ensure_document_index_ready(bound_document)
            excel_request = build_excel_answer_request(
                question=question,
                conversation_history=history_messages,
                document_id=int(bound_document["id"]),
                base_url=DEFAULT_BASE_URL,
                retrieval_backend=payload.retrieval_backend,
            )
            if not excel_request:
                raise ValueError("当前 Excel 中未检索到相关政策片段。")
        elif not bound_document:
            excel_request = build_excel_answer_request(
                question=question,
                conversation_history=history_messages,
                document_id=None,
                base_url=DEFAULT_BASE_URL,
                retrieval_backend=payload.retrieval_backend,
            )

        if excel_request:
            client = excel_request["client"]
            messages = excel_request["messages"]
            page_images: list[dict[str, Any]] = []
            routed_documents = excel_request["routed_documents"] or ([bound_document] if bound_document else [])
            answer_sources = excel_request["answer_sources"]
            request_kind = "excel"
            policies = excel_request["policies"]
        else:
            client, messages, page_images, routed_documents, answer_sources = build_chat_request_multi(
                question=question,
                conversation_history=history_messages,
                base_url=DEFAULT_BASE_URL,
                system_prompt=DOCUMENT_ASSISTANT_SYSTEM_PROMPT,
            )
            answer_sources = attach_answer_source_urls(answer_sources)
            request_kind = "pdf"
            policies = []
        logger.info(
            "[chat.sync.context] conversation_id=%s kind=%s routed_documents=%s page_images=%s",
            conversation_id,
            request_kind,
            [document["id"] for document in routed_documents],
            len(page_images),
        )

        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            extra_body={"enable_thinking": payload.enable_thinking},
        )
        result = {
            "answer": extract_response_text(response.choices[0].message.content),
            "usage": usage_to_dict(response.usage),
            "summary": (
                build_excel_summary(DEFAULT_MODEL, usage_to_dict(response.usage), policies)
                if request_kind == "excel"
                else build_summary(
                    model=DEFAULT_MODEL,
                    page_images=page_images,
                    usage=usage_to_dict(response.usage),
                    max_tokens=2048,
                )
            ),
        }
        logger.info(
            "[chat.sync.done] conversation_id=%s answer_len=%s usage=%s summary=%s",
            conversation_id,
            len(result["answer"]),
            result["usage"],
            result["summary"],
        )
    except EnvironmentError as exc:
        logger.exception(
            "[chat.sync.error] conversation_id=%s environment_error=%s",
            conversation_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "[chat.sync.error] conversation_id=%s request_failed=%s",
            conversation_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=f"模型请求失败：{exc}") from exc

    conversation, user_message, assistant_message = persist_round(
        conversation_id=conversation_id,
        document=routed_documents[0] if routed_documents else None,
        routed_documents=routed_documents,
        question=question,
        answer=result["answer"],
        answer_sources=answer_sources,
    )

    return {
        "conversation": conversation,
        "document": attach_pdf_url(routed_documents[0]) if routed_documents else None,
        "routed_documents": attach_pdf_urls(routed_documents),
        "answer_sources": answer_sources,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "usage": result["usage"],
        "summary": result["summary"],
    }
