import unittest
from unittest.mock import patch

from pdf_qa import retrieve_multi_document_context, score_keyword_match_pages


class PdfQaRetrievalTests(unittest.TestCase):
    def test_keyword_match_pages_is_case_insensitive_for_latin_terms(self):
        pages = {
            1: "The release process mentions on-hold tag status.",
            2: "Attachment A-1\nON-HOLD CARD\nMRB NUMBER\nON HOLD REASON",
        }
        query_plan = {"keywords": ["on-hold", "card"]}

        ranked = score_keyword_match_pages(
            "找一下 on-hold card",
            query_plan=query_plan,
            page_text_map=pages,
            top_k=2,
        )

        self.assertEqual(ranked[0], 2)

    def test_single_routed_document_keeps_up_to_thirty_pages(self):
        document = {
            "id": 14,
            "file_name": "guide.pdf",
            "display_name": "Guide",
            "page_count": 40,
            "pdf_url": "/pdf-files/guide.pdf",
            "render_dir": "/tmp/renders",
        }
        pages = list(range(1, 36))

        with patch("pdf_qa.retrieve_pages", return_value=pages) as retrieve_pages:
            context = retrieve_multi_document_context(
                question="我想看防安全执法检查专项行动统计表",
                routed_documents=[document],
            )

        retrieve_pages.assert_called_once_with(
            question="我想看防安全执法检查专项行动统计表",
            document_id=14,
            total_pages=40,
            max_pages=30,
        )
        self.assertEqual([item["page_number"] for item in context], list(range(1, 31)))

    def test_multiple_routed_documents_keep_per_document_limit(self):
        documents = [
            {
                "id": 1,
                "file_name": "a.pdf",
                "display_name": "A",
                "page_count": 40,
                "pdf_url": "/pdf-files/a.pdf",
                "render_dir": "/tmp/a",
            },
            {
                "id": 2,
                "file_name": "b.pdf",
                "display_name": "B",
                "page_count": 40,
                "pdf_url": "/pdf-files/b.pdf",
                "render_dir": "/tmp/b",
            },
        ]

        def fake_retrieve_pages(*, document_id: int, **kwargs):
            return list(range(document_id * 100, document_id * 100 + 10))

        with patch("pdf_qa.retrieve_pages", side_effect=fake_retrieve_pages) as retrieve_pages:
            context = retrieve_multi_document_context(
                question="比较两个文件",
                routed_documents=documents,
            )

        self.assertEqual(retrieve_pages.call_count, 2)
        self.assertEqual([item["page_number"] for item in context], [100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205])


if __name__ == "__main__":
    unittest.main()
