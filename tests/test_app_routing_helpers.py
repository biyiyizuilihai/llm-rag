import unittest
from unittest.mock import patch

from app import (
    HYBRID_RETRIEVAL_PROMPT_APPENDIX,
    build_global_chat_request,
    has_excel_retrieval_context,
    merge_excel_context_into_pdf_messages,
)


class AppRoutingHelperTests(unittest.TestCase):
    def test_empty_excel_preflight_has_no_retrieval_context(self):
        self.assertFalse(has_excel_retrieval_context(None))
        self.assertFalse(has_excel_retrieval_context({"policies": []}))

    def test_excel_preflight_with_policies_has_retrieval_context(self):
        self.assertTrue(has_excel_retrieval_context({"policies": [{"policy_id": 1}]}))

    def test_merge_excel_context_into_pdf_messages_appends_excel_evidence(self):
        messages = [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "PDF intro"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
                    {"type": "text", "text": "用户问题：查一下特采的流程"},
                ],
            },
        ]
        excel_request = {
            "policies": [{"policy_id": 1}],
            "messages": [
                {"role": "system", "content": "excel system"},
                {"role": "user", "content": "Excel policy context"},
            ]
        }

        merged = merge_excel_context_into_pdf_messages(messages, excel_request)

        user_content = merged[-1]["content"]
        self.assertEqual(user_content[-2]["type"], "text")
        self.assertIn("Excel policy context", user_content[-2]["text"])
        self.assertEqual(user_content[-1]["text"], "用户问题：查一下特采的流程")

    def test_global_chat_request_merges_excel_and_pdf_results(self):
        excel_request = {
            "client": object(),
            "messages": [{"role": "user", "content": "Excel policy context"}],
            "policies": [{"policy_id": 1}],
            "routed_documents": [{"id": 11, "file_type": "excel"}],
            "answer_sources": [{"type": "excel_policy"}],
        }
        pdf_client = object()
        pdf_messages = [
            {"role": "system", "content": "pdf system"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "PDF intro"},
                    {"type": "text", "text": "用户问题：查一下特采的流程"},
                ],
            },
        ]
        pdf_pages = [{"document_id": 6, "page_number": 22}]
        pdf_docs = [{"id": 6, "file_type": "pdf"}]
        pdf_sources = [{"document_id": 6, "pages": [22]}]

        with patch("app.build_excel_answer_request", return_value=excel_request), patch(
            "app.build_chat_request_multi",
            return_value=(pdf_client, pdf_messages, pdf_pages, pdf_docs, pdf_sources),
        ):
            result = build_global_chat_request(
                question="查一下特采的流程",
                conversation_history=[],
                base_url="https://example.test/v1",
            )

        self.assertEqual(result["request_kind"], "hybrid")
        self.assertIs(result["client"], pdf_client)
        self.assertEqual(result["page_images"], pdf_pages)
        self.assertEqual(result["policies"], excel_request["policies"])
        self.assertEqual(result["routed_documents"], [*pdf_docs, *excel_request["routed_documents"]])
        self.assertEqual(result["answer_sources"], [*pdf_sources, *excel_request["answer_sources"]])
        self.assertIn(HYBRID_RETRIEVAL_PROMPT_APPENDIX, result["messages"][0]["content"])
        self.assertIn("Excel policy context", result["messages"][-1]["content"][-2]["text"])


if __name__ == "__main__":
    unittest.main()
