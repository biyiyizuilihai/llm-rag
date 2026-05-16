import unittest
from unittest.mock import patch

import app


class AppEndpointTests(unittest.TestCase):
    def test_ocr_status_returns_latest_document_snapshot(self):
        document = {
            "id": 20,
            "file_name": "CRQA100H07_CN.pdf",
            "ocr_status": "done",
            "ocr_progress": 100,
            "ocr_detail": "索引完成",
            "chapter_summary": "修订历史概览（第161-183页）：覆盖完整文档末尾。",
        }

        with (
            patch("app.get_document", return_value=document),
            patch("app.get_ocr_status", return_value="done"),
            patch("app.attach_pdf_url", side_effect=lambda item: item | {"pdf_url": "/pdf-files/x.pdf"}),
        ):
            result = app.ocr_status(20)

        self.assertEqual(result["document"]["chapter_summary"], document["chapter_summary"])
        self.assertEqual(result["document"]["pdf_url"], "/pdf-files/x.pdf")


if __name__ == "__main__":
    unittest.main()
