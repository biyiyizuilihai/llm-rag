import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pdf_qa
from pdf_qa import (
    build_document_user_message,
    filter_documents_by_relevance,
    retrieve_multi_document_context,
    retrieve_pages,
    route_documents,
    score_keyword_match_pages,
)


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

        query_plan = {"scope": "targeted", "query_variants": ["专项行动统计表"]}
        with (
            patch("pdf_qa.understand_retrieval_query", return_value=query_plan),
            patch("pdf_qa.retrieve_pages", return_value=pages) as retrieve_pages_mock,
        ):
            context = retrieve_multi_document_context(
                question="我想看防安全执法检查专项行动统计表",
                routed_documents=[document],
            )

        retrieve_pages_mock.assert_called_once_with(
            question="我想看防安全执法检查专项行动统计表",
            document_id=14,
            total_pages=40,
            allow_fallback=True,
            query_plan=query_plan,
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

        query_plan = {"scope": "targeted", "query_variants": ["比较两个文件"]}
        with (
            patch("pdf_qa.understand_retrieval_query", return_value=query_plan),
            patch("pdf_qa.filter_documents_by_relevance", side_effect=lambda question, doc_pages, base_url="": doc_pages),
            patch("pdf_qa.retrieve_pages", side_effect=fake_retrieve_pages) as retrieve_pages_mock,
        ):
            context = retrieve_multi_document_context(
                question="比较两个文件",
                routed_documents=documents,
            )

        self.assertEqual(retrieve_pages_mock.call_count, 2)
        for call_args in retrieve_pages_mock.call_args_list:
            self.assertIs(call_args.kwargs["query_plan"], query_plan)
            self.assertFalse(call_args.kwargs["allow_fallback"])
            self.assertEqual(call_args.kwargs["max_pages"], 30)
        self.assertEqual([item["page_number"] for item in context], [100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205])

    def test_multi_document_filters_after_page_retrieval(self):
        documents = [
            {
                "id": 1,
                "file_name": "wrong.pdf",
                "display_name": "Wrong",
                "page_count": 40,
                "pdf_url": "/pdf-files/wrong.pdf",
                "render_dir": "/tmp/wrong",
            },
            {
                "id": 2,
                "file_name": "right.pdf",
                "display_name": "Right",
                "page_count": 40,
                "pdf_url": "/pdf-files/right.pdf",
                "render_dir": "/tmp/right",
            },
        ]
        query_plan = {"scope": "targeted", "query_variants": ["MRB trigger timing"]}

        def fake_retrieve_pages(*, document_id: int, **kwargs):
            return [document_id * 10 + 1, document_id * 10 + 2]

        def fake_filter(question, doc_pages, base_url=""):
            return [(document, pages) for document, pages in doc_pages if int(document["id"]) == 2]

        with (
            patch("pdf_qa.understand_retrieval_query", return_value=query_plan) as understand,
            patch("pdf_qa.retrieve_pages", side_effect=fake_retrieve_pages),
            patch("pdf_qa.filter_documents_by_relevance", side_effect=fake_filter) as relevance_filter,
        ):
            context = retrieve_multi_document_context(
                question="mrb触发时机",
                routed_documents=documents,
            )

        understand.assert_called_once_with("mrb触发时机")
        relevance_filter.assert_called_once()
        self.assertEqual([item["document_id"] for item in context], [2, 2])
        self.assertEqual([item["page_number"] for item in context], [21, 22])

    def test_route_documents_keeps_all_ready_documents_without_profile_prefilter(self):
        documents = [
            {"id": 3, "file_name": "c.pdf", "display_name": "C", "ocr_status": "done"},
            {"id": 2, "file_name": "b.pdf", "display_name": "B", "ocr_status": "done"},
            {"id": 1, "file_name": "a.pdf", "display_name": "A", "ocr_status": "done"},
        ]

        with (
            patch("pdf_qa.find_explicit_document_matches", return_value=[]),
            patch("pdf_qa.list_route_ready_documents", return_value=documents),
            patch("pdf_qa.understand_retrieval_query") as understand,
        ):
            routed = route_documents("mrb触发时机")

        understand.assert_not_called()
        self.assertEqual([document["id"] for document in routed], [3, 2, 1])

    def test_retrieve_pages_can_disable_sparse_fallback(self):
        with (
            patch("pdf_qa.get_page_ocr_text_map", return_value={}),
            patch("pdf_qa.fts_search_pages", return_value=[]),
            patch("pdf_qa.phrase_search_pages", return_value=[]),
            patch("pdf_qa.score_keyword_match_pages", return_value=[]),
            patch("pdf_qa.encode_texts", return_value=[]),
            patch("pdf_qa.collect_sparse_fallback_pages", return_value=[1, 2, 3]) as fallback,
        ):
            pages = retrieve_pages(
                question="mrb触发时机",
                document_id=9,
                total_pages=20,
                allow_fallback=False,
                query_plan={"scope": "targeted", "query_variants": ["MRB trigger timing"]},
            )

        fallback.assert_not_called()
        self.assertEqual(pages, [])

    def test_document_relevance_filter_fails_open_when_llm_unavailable(self):
        doc_pages = [
            ({"id": 1, "file_name": "a.pdf", "display_name": "A"}, [1, 2]),
            ({"id": 2, "file_name": "b.pdf", "display_name": "B"}, [3, 4]),
        ]

        with patch("pdf_qa.build_openai_client", side_effect=EnvironmentError("missing key")):
            filtered = filter_documents_by_relevance("mrb触发时机", doc_pages)

        self.assertIs(filtered, doc_pages)

    def test_document_relevance_filter_uses_chapter_summary(self):
        doc_pages = [
            (
                {
                    "id": 1,
                    "file_name": "wrong.pdf",
                    "display_name": "Wrong",
                    "summary_text": "generic quality document",
                    "chapter_summary": "总则说明（第1-3页）：无 MRB 内容。",
                },
                [1, 2],
            ),
            (
                {
                    "id": 2,
                    "file_name": "right.pdf",
                    "display_name": "Right",
                    "summary_text": "generic quality document",
                    "chapter_summary": "MRB 触发时机（第21-24页）：说明 MRB 的触发条件。",
                },
                [21, 22],
            ),
        ]
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured["user_message"] = kwargs["messages"][1]["content"]
                message = SimpleNamespace(content='{"relevant_document_ids": [2]}')
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        with patch("pdf_qa.build_openai_client", return_value=fake_client):
            filtered = filter_documents_by_relevance("mrb触发时机", doc_pages)

        self.assertIn("内容分布", captured["user_message"])
        self.assertIn("MRB 触发时机（第21-24页）", captured["user_message"])
        self.assertEqual([document["id"] for document, _ in filtered], [2])

    def test_build_chapter_summary_saves_page_range_summary(self):
        fake_document = {"id": 7, "file_name": "manual.pdf", "page_count": 2}
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured["user_message"] = kwargs["messages"][1]["content"]
                message = SimpleNamespace(content="MRB 触发时机（第1-2页）：说明触发条件。")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        with (
            patch("pdf_qa.get_document", return_value=fake_document),
            patch("pdf_qa.get_page_ocr_text_map", return_value={1: "MRB trigger", 2: "timing"}),
            patch("pdf_qa.build_openai_client", return_value=fake_client),
            patch("pdf_qa.update_chapter_summary") as update_summary,
        ):
            summary = pdf_qa.build_chapter_summary(7)

        self.assertIn("[第1页]", captured["user_message"])
        self.assertIn("[第2页]", captured["user_message"])
        self.assertEqual(summary, "MRB 触发时机（第1-2页）：说明触发条件。")
        update_summary.assert_called_once_with(7, summary)

    def test_build_chapter_summary_allows_long_page_range_output(self):
        fake_document = {"id": 8, "file_name": "long.pdf", "page_count": 168}
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured["max_tokens"] = kwargs["max_tokens"]
                message = SimpleNamespace(content="整体内容（第1-168页）：覆盖完整文档。")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        with (
            patch("pdf_qa.get_document", return_value=fake_document),
            patch("pdf_qa.get_page_ocr_text_map", return_value={page: f"page {page}" for page in range(1, 169)}),
            patch("pdf_qa.build_openai_client", return_value=fake_client),
            patch("pdf_qa.update_chapter_summary"),
        ):
            pdf_qa.build_chapter_summary(8)

        self.assertGreaterEqual(captured["max_tokens"], 4096)

    def test_build_chapter_summary_summarizes_long_documents_in_page_chunks(self):
        fake_document = {"id": 9, "file_name": "chunked.pdf", "page_count": 85}
        captured_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                user_message = kwargs["messages"][1]["content"]
                captured_messages.append(user_message)
                if "[第1页]" in user_message:
                    content = "前段内容（第1-40页）：说明前段。"
                elif "[第41页]" in user_message:
                    content = "中段内容（第41-80页）：说明中段。"
                else:
                    content = "后段内容（第81-85页）：说明后段。"
                message = SimpleNamespace(content=content)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        with (
            patch.object(pdf_qa, "DOCUMENT_CHAPTER_SUMMARY_CHUNK_SIZE", 40, create=True),
            patch("pdf_qa.get_document", return_value=fake_document),
            patch("pdf_qa.get_page_ocr_text_map", return_value={page: f"page {page}" for page in range(1, 86)}),
            patch("pdf_qa.build_openai_client", return_value=fake_client),
            patch("pdf_qa.update_chapter_summary") as update_summary,
        ):
            summary = pdf_qa.build_chapter_summary(9)

        self.assertEqual(len(captured_messages), 3)
        self.assertIn("[第40页]", captured_messages[0])
        self.assertNotIn("[第41页]", captured_messages[0])
        self.assertIn("[第41页]", captured_messages[1])
        self.assertIn("[第85页]", captured_messages[2])
        self.assertIn("前段内容（第1-40页）", summary)
        self.assertIn("中段内容（第41-80页）", summary)
        self.assertIn("后段内容（第81-85页）", summary)
        update_summary.assert_called_once_with(9, summary)

    def test_multi_document_message_does_not_claim_single_pdf(self):
        message = build_document_user_message(
            page_images=[
                {
                    "document_id": 1,
                    "document_name": "a.pdf",
                    "document_display_name": "A",
                    "page_number": 1,
                    "data_url": "data:image/jpeg;base64,a",
                },
                {
                    "document_id": 2,
                    "document_name": "b.pdf",
                    "document_display_name": "B",
                    "page_number": 2,
                    "data_url": "data:image/jpeg;base64,b",
                },
            ],
            question="mrb触发时机",
        )

        intro = message["content"][0]["text"]
        self.assertIn("多个 PDF", intro)
        self.assertNotIn("唯一需要参考的文档", intro)


if __name__ == "__main__":
    unittest.main()
