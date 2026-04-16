# LLM RAG PDF QA

一个面向 PDF 的本地文档问答系统，支持 OCR 建索引、页级混合检索、多文档路由、流式回答和 PDF 原文预览。

## Features

- 上传 PDF 后自动渲染页图并异步构建索引
- 基于 PaddleOCR 异步接口进行整页 OCR
- SQLite FTS + 向量检索的页级混合召回
- 多文档会话与文档级粗路由
- 仅发送少量命中页给模型，避免整本送模
- 流式回答、思考过程展示、PDF.js 原文跳转
- OCR 大文件分段处理、失败重试、后台队列

## Stack

- Backend: FastAPI
- Storage: SQLite + FTS5 + sqlite-vec
- OCR: PaddleOCR async API
- Embedding: SiliconFlow `BAAI/bge-m3`
- LLM: DashScope compatible API
- PDF rendering: PyMuPDF + Pillow
- Frontend: Vanilla HTML / CSS / JavaScript
- Viewer: PDF.js

## Project Structure

```text
app.py                FastAPI routes and chat APIs
pdf_qa.py             OCR, retrieval, multi-document routing, LLM request building
storage.py            SQLite schema and storage helpers
static/               Frontend assets and PDF.js viewer
data/                 Runtime data directory (ignored by git)
```

## Requirements

- Python 3.10+
- DashScope API key
- PaddleOCR async API token
- SiliconFlow API key

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

填写 `.env` 中的实际配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
PADDLEOCR_TOKEN=your_paddleocr_token
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

## Run

开发模式：

```bash
uvicorn app:app --reload
```

生产模式：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

默认访问地址：

```text
http://127.0.0.1:8000
```

## Retrieval Flow

1. 上传 PDF
2. 渲染页图并提交 OCR 任务
3. 保存页级 OCR 文本
4. 建立 FTS 和向量索引
5. 生成文档画像用于文档级路由
6. 提问时先做 query understanding
7. 文档级粗路由选出候选文档
8. 候选文档内执行页级混合检索
9. 仅发送命中页图片给模型回答

## OCR Notes

- 默认后台 OCR 并发为 `1`
- 大 PDF 会按页数自动分段 OCR
- 对 `429/500/502/503/504` 和网络错误会自动重试
- 运行时数据、上传文件、渲染缓存、数据库默认不入库

## Environment Variables

核心变量：

- `DASHSCOPE_API_KEY`
- `PADDLEOCR_TOKEN`
- `SILICONFLOW_API_KEY`

可选变量：

- `PADDLEOCR_MODEL`
- `PADDLEOCR_JOB_URL`
- `SILICONFLOW_EMBEDDING_URL`
- `SILICONFLOW_EMBEDDING_MODEL`
- `SILICONFLOW_EMBEDDING_BATCH_SIZE`
- `OCR_PIPELINE_MAX_WORKERS`
- `PADDLEOCR_RETRY_ATTEMPTS`
- `PADDLEOCR_RETRY_BACKOFF`
- `PADDLEOCR_STALL_TIMEOUT`
- `PADDLEOCR_JOB_TIMEOUT`
- `PADDLEOCR_CHUNK_PAGE_THRESHOLD`
- `PADDLEOCR_CHUNK_SIZE`

## Notes

- `data/` 下的数据库、上传文件和渲染缓存属于运行时数据，不在仓库中保存
- 本项目默认面向本地或内网部署
- 如果 OCR 或模型服务密钥更新，需要重启服务

## License

Internal / private project use only.
