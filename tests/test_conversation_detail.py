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


if __name__ == "__main__":
    unittest.main()
