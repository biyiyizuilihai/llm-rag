import base64
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


LANGGRAPH_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LANGGRAPH_DIR.parent


def import_isolated_tool():
    sys.modules.pop("pdf_retrieval_tool", None)
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != PROJECT_ROOT.resolve()
    ]
    if str(LANGGRAPH_DIR) not in sys.path:
        sys.path.insert(0, str(LANGGRAPH_DIR))
    import pdf_retrieval_tool

    return pdf_retrieval_tool


def create_test_database(data_dir: Path, *, status: str = "done") -> None:
    uploads_dir = data_dir / "uploads"
    render_dir = data_dir / "renders" / "doc-render"
    uploads_dir.mkdir(parents=True)
    render_dir.mkdir(parents=True)
    (uploads_dir / "policy.pdf").write_bytes(b"%PDF-1.4\n")

    image = Image.new("RGB", (24, 24), color=(255, 255, 255))
    image.save(render_dir / "page-0003.jpg", format="JPEG")
    image.save(render_dir / "page-0005.jpg", format="JPEG")

    conn = sqlite3.connect(data_dir / "app.db")
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            file_sha256 TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            render_dir TEXT NOT NULL,
            page_count INTEGER NOT NULL,
            version_index INTEGER NOT NULL,
            file_type TEXT NOT NULL DEFAULT 'pdf',
            row_count INTEGER NOT NULL DEFAULT 0,
            config_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            ocr_status TEXT NOT NULL DEFAULT 'pending',
            ocr_progress INTEGER NOT NULL DEFAULT 0,
            ocr_detail TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE page_ocr (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            ocr_text TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO documents (
            id, file_name, display_name, file_sha256, storage_path, render_dir,
            page_count, version_index, file_type, created_at, updated_at,
            ocr_status, ocr_progress, ocr_detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            7,
            "policy.pdf",
            "policy",
            "sha",
            str(uploads_dir / "policy.pdf"),
            str(render_dir),
            10,
            1,
            "pdf",
            "now",
            "now",
            status,
            35 if status != "done" else 100,
            "正在索引" if status != "done" else "完成",
        ),
    )
    conn.executemany(
        "INSERT INTO page_ocr (id, document_id, page_number, ocr_text) VALUES (?, ?, ?, ?)",
        [
            (1, 7, 3, "第三页文本 补贴标准"),
            (2, 7, 5, "第五页文本 申报条件"),
        ],
    )
    conn.commit()
    conn.close()


class PdfRetrievalToolTests(unittest.TestCase):
    def test_find_pdf_document_by_name_returns_best_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            create_test_database(data_dir)
            os.environ["LLM_RAG_DATA_DIR"] = str(data_dir)
            pdf_retrieval_tool = import_isolated_tool()

            payload = pdf_retrieval_tool.find_pdf_document_payload("policy")

        result = json.loads(payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["document"]["id"], 7)
        self.assertEqual(result["document"]["display_name"], "policy")

    def test_find_pdf_document_by_spaced_name_returns_best_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            create_test_database(data_dir)
            os.environ["LLM_RAG_DATA_DIR"] = str(data_dir)
            pdf_retrieval_tool = import_isolated_tool()

            payload = pdf_retrieval_tool.find_pdf_document_payload("policy pdf")

        result = json.loads(payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["document"]["id"], 7)

    def test_retrieve_pdf_pages_payload_accepts_document_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            create_test_database(data_dir)
            os.environ["LLM_RAG_DATA_DIR"] = str(data_dir)
            pdf_retrieval_tool = import_isolated_tool()

            payload = pdf_retrieval_tool.retrieve_pdf_pages_payload(
                question="补贴标准是什么？",
                document_name="policy",
                max_pages=1,
            )

        result = json.loads(payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["document_id"], 7)
        self.assertEqual(result["page_numbers"], [3])

    def test_retrieve_pdf_pages_payload_returns_hits_with_images_and_text_without_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            create_test_database(data_dir)
            os.environ["LLM_RAG_DATA_DIR"] = str(data_dir)
            pdf_retrieval_tool = import_isolated_tool()

            payload = pdf_retrieval_tool.retrieve_pdf_pages_payload(
                question="补贴标准是什么？",
                document_id=7,
                max_pages=2,
            )

        result = json.loads(payload)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["document_id"], 7)
        self.assertEqual(result["question"], "补贴标准是什么？")
        self.assertEqual([hit["page_number"] for hit in result["hits"]], [3, 5])
        self.assertEqual(result["hits"][0]["ocr_text"], "第三页文本 补贴标准")
        self.assertTrue(result["hits"][0]["image_data_url"].startswith("data:image/jpeg;base64,"))
        base64.b64decode(result["hits"][0]["image_data_url"].split(",", 1)[1])

    def test_retrieve_pdf_pages_payload_rejects_unready_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            create_test_database(data_dir, status="processing")
            os.environ["LLM_RAG_DATA_DIR"] = str(data_dir)
            pdf_retrieval_tool = import_isolated_tool()

            payload = pdf_retrieval_tool.retrieve_pdf_pages_payload(
                question="这是什么？",
                document_id=7,
            )

        result = json.loads(payload)
        self.assertEqual(result["status"], "not_ready")
        self.assertEqual(result["ocr_status"], "processing")
        self.assertEqual(result["ocr_progress"], 35)


if __name__ == "__main__":
    unittest.main()
