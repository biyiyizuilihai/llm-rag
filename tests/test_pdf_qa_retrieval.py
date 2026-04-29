import unittest

from pdf_qa import score_keyword_match_pages


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


if __name__ == "__main__":
    unittest.main()
