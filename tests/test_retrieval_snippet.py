import unittest

from rag_indexer.retrieval import _make_snippet


class TestRetrievalSnippet(unittest.TestCase):
    def test_snippet_prefers_query_match_zone(self):
        text = (
            "intro " * 40
            + "criticalToken lives here "
            + "tail " * 40
        )
        snippet = _make_snippet(text, max_len=80, query_text="where is criticalToken used")

        self.assertIn("criticalToken", snippet)
        self.assertTrue(snippet.startswith("...") or len(text) <= 80)

    def test_snippet_falls_back_to_prefix_without_match(self):
        text = "a" * 500
        snippet = _make_snippet(text, max_len=60, query_text="invoice")

        self.assertEqual(snippet, ("a" * 57) + "...")


if __name__ == "__main__":
    unittest.main()
