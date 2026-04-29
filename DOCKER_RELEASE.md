# Docker Release

This package can run with the bundled `data/` directory. The SQLite database
contains the existing document metadata, OCR text, Excel chunks, and sqlite-vec
vector indexes.

## Start

```bash
cp .env.example .env
# Edit .env and fill LLM_API_KEY. Fill SILICONFLOW_API_KEY/PADDLEOCR_TOKEN only
# if you need to upload or rebuild indexes.
docker compose up --build -d
```

Open:

```text
http://SERVER_IP:8000
```

## Data

Runtime data is mounted from:

```text
./data:/app/data
```

Do not delete `data/app.db` if you want to keep the bundled Excel/PDF indexes.
The app rewrites stored document paths on startup so a database created on
another machine can run inside the container.

## Common Commands

```bash
docker compose logs -f
docker compose restart
docker compose down
```
