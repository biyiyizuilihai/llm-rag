import unittest
from unittest.mock import patch

import app


class ConversationDetailTests(unittest.TestCase):
    def test_global_conversation_restores_primary_routed_pdf_document(self):
        routed_document = {
            "id": 9,
            "file_name": "CRQA100H07.pdf",
            "file_type": "pdf",
            "storage_filename": "CRQA100H07.pdf",
        }

        with (
            patch("app.get_conversation", return_value={"id": 3, "document_id": None}),
            patch("app.get_document") as get_document,
            patch("app.list_messages", return_value=[]),
            patch("app.list_conversation_documents", return_value=[routed_document]),
            patch(
                "app.attach_pdf_url",
                side_effect=lambda item: None
                if item is None
                else item | {"pdf_url": f"/pdf-files/{item['storage_filename']}"},
            ),
        ):
            result = app.conversation_detail(3)

        get_document.assert_not_called()
        self.assertEqual(result["document"]["id"], routed_document["id"])
        self.assertEqual(result["document"]["pdf_url"], "/pdf-files/CRQA100H07.pdf")
        self.assertEqual(result["routed_documents"][0]["id"], routed_document["id"])

    def test_global_conversation_prefers_latest_answer_source_document_for_preview(self):
        routed_documents = [
            {
                "id": 20,
                "file_name": "CRQA100H07_CN.pdf",
                "file_type": "pdf",
                "storage_filename": "CRQA100H07_CN.pdf",
            },
            {
                "id": 19,
                "file_name": "申报材料_01.江苏省先进级智能工厂申报书-盖章版.pdf",
                "file_type": "pdf",
                "storage_filename": "申报材料.pdf",
            },
        ]
        messages = [
            {"id": 1, "role": "user", "content": "我想问米巴精密的实例清单"},
            {
                "id": 2,
                "role": "assistant",
                "content": "第 5 页",
                "metadata": {
                    "answer_sources": [
                        {
                            "document_id": 19,
                            "pages": [5],
                        }
                    ]
                },
            },
        ]

        def get_document(document_id):
            return routed_documents[1] if document_id == 19 else None

        with (
            patch("app.get_conversation", return_value={"id": 179, "document_id": None}),
            patch("app.get_document", side_effect=get_document),
            patch("app.list_messages", return_value=messages),
            patch("app.list_conversation_documents", return_value=routed_documents),
            patch(
                "app.attach_pdf_url",
                side_effect=lambda item: None
                if item is None
                else item | {"pdf_url": f"/pdf-files/{item['storage_filename']}"},
            ),
        ):
            result = app.conversation_detail(179)

        self.assertEqual(result["document"]["id"], 19)
        self.assertEqual(result["document"]["pdf_url"], "/pdf-files/申报材料.pdf")
        self.assertEqual([document["id"] for document in result["routed_documents"]], [20, 19])


if __name__ == "__main__":
    unittest.main()
