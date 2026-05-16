# LLM RAG Document QA

这是一个本地文档问答工作台，后端使用 FastAPI，前端使用原生 HTML/CSS/JavaScript。它可以上传 PDF 或 Excel 文件，自动建立本地检索索引，然后通过大模型做带来源的问答。

当前代码支持两条主要链路：

- PDF 文档问答：PDF 上传、页面渲染、PaddleOCR 异步识别、页级 FTS/向量检索、多文档路由、源页跳转。
- Excel 政策库问答：Excel 上传、字段配置、政策行切分、FTS/向量检索、按政策来源回答。

## 功能概览

- 支持上传 `.pdf`、`.xls`、`.xlsx`、`.xlsm`。
- PDF 上传后自动渲染页面图片，并在后台提交 OCR/索引任务。
- Excel 上传后先预览字段，确认标题字段、正文字段、过滤字段和来源字段后再建立索引。
- SQLite 保存文档、会话、消息、OCR 文本、Excel 政策块、FTS 索引和可选向量索引。
- SQLite FTS5 + `sqlite-vec` 组成混合检索；如果向量不可用，会尽量回退到全文检索。
- 全局问答会并行检索 PDF 和 Excel，再让模型判断使用哪些证据。
- 流式返回模型回答，支持 reasoning 内容、回答正文、token 用量和来源元数据。
- PDF 答案来源可跳转到原 PDF 对应页。
- 支持 Docker Compose 本地部署，也支持直接用 Python 虚拟环境运行。

## 技术栈

- Backend: FastAPI + Uvicorn
- Frontend: Vanilla HTML / CSS / JavaScript
- Database: SQLite
- Full-text Search: SQLite FTS5
- Vector Search: `sqlite-vec`
- PDF: PyMuPDF + Pillow + PDF.js
- Excel: `openpyxl` / `xlrd`
- LLM: OpenAI-compatible API，默认面向 DashScope 兼容接口
- Embedding: SiliconFlow `BAAI/bge-m3`
- OCR: PaddleOCR async API

## 目录结构

```text
.
├── app.py                  # FastAPI 入口、API 路由、上传/会话/流式问答编排
├── pdf_qa.py               # PDF 渲染、OCR、页级检索、文档路由、LLM 请求构造
├── excel_qa.py             # Excel 预览、字段配置、政策切块、Excel 检索和问答上下文
├── storage.py              # SQLite 表结构、迁移、FTS/vector、文档/会话/消息持久化
├── static/                 # 原生前端资源
│   ├── index.html
│   ├── app.css
│   ├── app.js
│   └── pdfjs/              # PDF.js 相关文件
├── docs/                   # 开发交接记录和设计笔记
├── tests/                  # unittest 测试
├── data/                   # 本地运行数据，包含数据库、上传文件和渲染缓存
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example            # 环境变量模板，不包含真实密钥
└── README.md
```

## 运行数据说明

默认运行数据目录是 `./data`，也可以通过 `LLM_RAG_DATA_DIR` 修改。

重要路径：

- `data/app.db`：当前代码默认使用的 SQLite 数据库。
- `data/uploads/`：用户上传的 PDF/Excel 原文件。
- `data/renders/`：PDF 页面渲染图和给模型使用的图片缓存。

这些文件可能包含真实业务文档、OCR 文本和问答记录，按敏感数据处理。不要把真实 `.env`、上传文件、运行数据库提交到公开仓库。

## 外部服务

这个项目本身是本地服务，但问答和索引会调用外部 API。

| 服务 | 用途 | 相关变量 |
| --- | --- | --- |
| LLM OpenAI-compatible API | 最终回答、PDF 查询理解、文档画像、内容分布摘要、相关性过滤 | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` |
| DashScope 兼容接口 | 默认 LLM 接口示例 | `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` |
| SiliconFlow Embedding | PDF/Excel 向量索引和向量检索 | `SILICONFLOW_API_KEY`, `SILICONFLOW_EMBEDDING_*` |
| PaddleOCR async API | PDF 页面 OCR | `PADDLEOCR_TOKEN`, `PADDLEOCR_*` |

最小可用配置通常至少需要：

```env
LLM_API_KEY=你的 DashScope 或兼容服务 API Key
SILICONFLOW_API_KEY=你的 SiliconFlow API Key
PADDLEOCR_TOKEN=你的 PaddleOCR Token
```

如果只是测试界面或已有索引数据，可以先只配置 `LLM_API_KEY`；但上传新 PDF、重建索引和高质量向量检索需要后两个服务。

## 本机部署：Python 虚拟环境

### 1. 准备环境

建议使用 Python 3.10 或更新版本。

```bash
cd /Users/biyiyi/Downloads/test-graphify/llm-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 创建 `.env`

```bash
cp .env.example .env
```

然后编辑 `.env`，至少填入你的真实 API Key。

```env
LLM_API_KEY=替换成你的 DashScope 或 OpenAI-compatible API Key
SILICONFLOW_API_KEY=替换成你的 SiliconFlow API Key
PADDLEOCR_TOKEN=替换成你的 PaddleOCR Token
```

不要把 `.env` 提交到 Git。仓库只应该提交 `.env.example` 这种不含真实密钥的模板文件。

### 3. 启动服务

开发模式：

```bash
uvicorn app:app --reload
```

指定端口：

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

浏览器打开：

```text
http://127.0.0.1:8000
```

### 4. 健康检查

```bash
curl http://127.0.0.1:8000/api/health
```

正常返回：

```json
{"status":"ok"}
```

## 本机部署：Docker Compose

Docker 方式适合希望一条命令启动服务的场景。

### 1. 准备 `.env`

```bash
cp .env.example .env
```

编辑 `.env`，填入真实 key。`APP_PORT` 决定宿主机访问端口，默认是 `8000`。

### 2. 启动

```bash
docker compose up --build -d
```

访问：

```text
http://127.0.0.1:8000
```

如果 `.env` 里设置了 `APP_PORT=8001`，则访问：

```text
http://127.0.0.1:8001
```

### 3. 查看日志

```bash
docker compose logs -f
```

### 4. 停止服务

```bash
docker compose down
```

Docker Compose 会把本机 `./data` 挂载到容器 `/app/data`。因此重启容器不会删除数据库和上传文件。

## `.env` 配置说明

完整模板见 [.env.example](.env.example)。下面是核心配置解释。

### 应用配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_PORT` | `8000` | Docker Compose 暴露到宿主机的端口。直接用 `uvicorn` 启动时，以命令行 `--port` 为准。 |
| `LLM_RAG_DATA_DIR` | `./data` | 数据目录，保存数据库、上传文件和渲染缓存。Docker 中会被覆盖为 `/app/data`。 |

### 大模型配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 空 | 推荐使用的主 LLM API Key。代码会优先读取它。 |
| `DASHSCOPE_API_KEY` | 空 | 旧配置兼容别名。如果没有 `LLM_API_KEY`，代码会读取它。 |
| `LLM_BASE_URL` | DashScope 兼容地址 | OpenAI-compatible API base URL。 |
| `LLM_MODEL` | `qwen3.5-35b-a3b` | 用于问答、查询理解、文档画像、内容分布摘要的模型名。 |

默认 DashScope 兼容地址：

```env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

如果你换成其他兼容 OpenAI Chat Completions 的服务，需要同时修改 `LLM_BASE_URL`、`LLM_MODEL` 和 `LLM_API_KEY`。

### 向量模型配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SILICONFLOW_API_KEY` | 空 | SiliconFlow API Key。缺失时新建向量索引会失败，部分检索会退化为 FTS。 |
| `SILICONFLOW_EMBEDDING_URL` | `https://api.siliconflow.cn/v1/embeddings` | Embedding API 地址。 |
| `SILICONFLOW_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding 模型。 |
| `SILICONFLOW_EMBEDDING_BATCH_SIZE` | `16` | 批量 embedding 数量。网络或服务不稳定时可调小。 |

### OCR 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PADDLEOCR_TOKEN` | 空 | PaddleOCR async API token。上传或重建 PDF 索引时必须配置。 |
| `PADDLEOCR_MODEL` | `PP-OCRv5` | OCR 模型名。 |
| `PADDLEOCR_JOB_URL` | PaddleOCR async job URL | 提交和轮询 OCR 任务的接口地址。 |
| `OCR_PIPELINE_MAX_WORKERS` | `1` | 后台索引任务并发数。建议本机先保持 1，避免 API 限流。 |
| `PADDLEOCR_RETRY_ATTEMPTS` | `3` | OCR 可重试错误的最大尝试次数。 |
| `PADDLEOCR_RETRY_BACKOFF` | `5` | OCR 重试基础等待秒数。第 N 次重试约等待 `5 * N` 秒。 |
| `PADDLEOCR_STALL_TIMEOUT` | `180` | OCR 进度长时间不变时判定卡住的秒数。 |
| `PADDLEOCR_JOB_TIMEOUT` | `1800` | 单个 OCR 任务最大等待秒数。 |
| `PADDLEOCR_CHUNK_PAGE_THRESHOLD` | `48` | PDF 页数超过该值时启用分段 OCR。 |
| `PADDLEOCR_CHUNK_SIZE` | `40` | 分段 OCR 每段页数。 |

### PDF 内容分布摘要配置

PDF OCR 后会生成“哪页到哪页主要讲什么”的内容分布摘要，用于文档库展示、路由和回答定位。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DOCUMENT_CHAPTER_SUMMARY_MAX_TOKENS` | `4096` | 每段内容分布摘要调用 LLM 时允许输出的最大 token。 |
| `DOCUMENT_CHAPTER_SUMMARY_CHUNK_SIZE` | `40` | 生成内容分布时，每次发送给 LLM 的页数范围。 |

### Excel 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EXCEL_QUERY_CLASSIFIER_MODEL` | 空 | 可选的小模型分类器。留空时使用规则解析用户问题。 |
| `EXCEL_VECTOR_INDEX_ENABLED` | `1` | Excel 是否写入向量索引。设置为 `0` 时只使用 FTS。 |

`EXCEL_QUERY_CLASSIFIER_MODEL` 可以填 `qwen-turbo`、`qwen-plus` 等轻量模型，用于把政策问题解析成关键词和过滤条件。留空也可以运行，只是过滤条件解析会更依赖规则。

## 使用流程

### PDF 问答

1. 打开首页。
2. 上传 PDF。
3. 等待索引完成。文档库会显示 OCR 状态、进度和索引详情。
4. 进入单文件问答或全局问答。
5. 提问后，系统会检索相关 PDF 页面，并把命中页图片和 OCR 文本交给模型。
6. 回答中的来源可跳转回原 PDF 页面。

PDF 索引过程：

```text
PDF 上传
→ PyMuPDF 渲染页图
→ PaddleOCR 异步 OCR
→ 保存页级 OCR 文本
→ 写入 SQLite FTS
→ 调用 SiliconFlow 生成页向量
→ 生成文档画像
→ 生成内容分布摘要
→ 文档状态变为 done
```

### Excel 政策库问答

1. 上传 Excel。
2. 前端展示字段预览。
3. 配置字段：
   - 标题字段：一条政策的名称。
   - 正文字段：进入检索和问答的主要内容。
   - 过滤字段：例如级别、地区、类型。
   - 来源字段：例如发文字号、来源行。
4. 提交配置后后台建立 Excel 索引。
5. 提问时系统检索相关政策片段，并要求模型按政策标题、发文字号或行号标注来源。

Excel 索引过程：

```text
Excel 上传
→ 解析表头和样例行
→ 用户确认字段配置
→ 每一行转换成 policy
→ 正文切成 chunks
→ 写入 SQLite FTS
→ 可选写入 sqlite-vec 向量索引
→ 文档状态变为 done
```

### 全局问答

全局问答会同时尝试：

- PDF 多文档路由和页级检索。
- Excel 政策库检索。

如果两边都有候选证据，系统会把两类证据合并给模型，并在系统提示中要求模型先判断来源是否相关，避免把不相干 PDF 和政策片段硬混在一起。

## 常用开发命令

运行测试：

```bash
python3 -m unittest discover -s tests
```

检查 Python 语法：

```bash
python3 -m py_compile app.py pdf_qa.py excel_qa.py storage.py
```

查看当前 Git 状态：

```bash
git status -sb
```

## API 简表

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/` | 前端首页 |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/bootstrap` | 获取文档和会话初始状态 |
| `POST` | `/api/documents` | 上传 PDF/Excel |
| `GET` | `/api/documents/{document_id}/ocr-status` | 查询索引状态 |
| `POST` | `/api/documents/{document_id}/rebuild-index` | 重建 PDF/Excel 索引 |
| `DELETE` | `/api/documents/{document_id}` | 删除文档和相关运行文件 |
| `GET` | `/api/documents/{document_id}/excel-preview` | 获取 Excel 预览 |
| `POST` | `/api/documents/{document_id}/excel-config` | 保存 Excel 字段配置并开始索引 |
| `POST` | `/api/conversations` | 创建会话 |
| `GET` | `/api/conversations/{conversation_id}` | 获取会话详情 |
| `DELETE` | `/api/conversations/{conversation_id}` | 删除会话 |
| `POST` | `/api/conversations/{conversation_id}/stream` | 流式问答 |
| `POST` | `/api/conversations/{conversation_id}/messages` | 非流式问答 |

## 常见问题

### 1. 启动后打不开页面

先检查服务是否在运行：

```bash
curl http://127.0.0.1:8000/api/health
```

如果端口被占用，换一个端口启动：

```bash
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

Docker 模式下可以改 `.env`：

```env
APP_PORT=8001
```

然后重启：

```bash
docker compose up --build -d
```

### 2. 上传 PDF 后索引失败

重点检查：

- `.env` 是否有 `PADDLEOCR_TOKEN`。
- PaddleOCR 服务地址 `PADDLEOCR_JOB_URL` 是否能访问。
- PDF 是否损坏或加密。
- 日志里是否有 API 限流、网络超时或 token 无效。

查看日志：

```bash
docker compose logs -f
```

或直接运行时查看终端输出。

### 3. 问答时报缺少 LLM key

代码会优先读取 `LLM_API_KEY`，没有时再读取 `DASHSCOPE_API_KEY`。

推荐写法：

```env
LLM_API_KEY=你的真实 API Key
DASHSCOPE_API_KEY=
```

改完 `.env` 后需要重启服务。

### 4. 检索效果不好或没有向量结果

检查：

- `SILICONFLOW_API_KEY` 是否有效。
- `SILICONFLOW_EMBEDDING_MODEL` 是否和已有向量维度一致。
- `EXCEL_VECTOR_INDEX_ENABLED` 是否被设置成 `0`。
- 上传或重建索引时是否成功生成向量。

没有向量时系统仍可能用 FTS 检索，但语义召回会变弱。

### 5. Excel 上传后需要配置字段

这是正常流程。Excel 不像 PDF 有固定页面结构，系统需要知道哪些列是标题、正文、过滤条件和来源字段。字段配置保存后才会建立政策库索引。

### 6. `.env` 改了但不生效

`.env` 在服务启动时读取。修改后需要重启：

```bash
# Python 直接启动时，停止 uvicorn 后重新运行
uvicorn app:app --reload

# Docker Compose
docker compose restart
```

## 安全注意事项

- 不要提交真实 `.env`。
- 不要把真实客户文档、上传文件、OCR 文本、运行数据库提交到公开仓库。
- `data/uploads/` 和 `data/renders/` 可能包含敏感信息。
- 对外网部署前应增加认证、访问控制、HTTPS、上传大小限制和日志脱敏；当前代码默认更适合本地或内网使用。

## 部署检查清单

上线或交给别人本机运行前，至少检查：

- [ ] 已创建 `.env`，并填写 `LLM_API_KEY`。
- [ ] 如果需要上传 PDF，已填写 `PADDLEOCR_TOKEN`。
- [ ] 如果需要高质量检索或重建向量索引，已填写 `SILICONFLOW_API_KEY`。
- [ ] `LLM_RAG_DATA_DIR` 指向希望保存数据的位置。
- [ ] `python3 -m unittest discover -s tests` 通过。
- [ ] 浏览器能打开首页。
- [ ] `/api/health` 返回 `{"status":"ok"}`。

## License

Internal / private project use only.
