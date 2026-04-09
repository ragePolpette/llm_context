import os
import unittest

from rag_indexer.db import get_connection
from rag_indexer.embedder import LocalHashEmbedder
from rag_indexer.retrieval import retrieve_v2
from rag_indexer.store import RagStore


class TestRetrievalV2(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("LLM_CONTEXT_TEST_DSN"), "LLM_CONTEXT_TEST_DSN not set"
    )
    def test_retrieve_v2_expected_path(self):
        dsn = os.getenv("LLM_CONTEXT_TEST_DSN")
        repo_id = os.getenv("LLM_CONTEXT_TEST_REPO", "myproj")
        expected = os.getenv("LLM_CONTEXT_TEST_EXPECTED_PATH")
        embedder = LocalHashEmbedder(dim=384)
        conn = get_connection(dsn)
        store = RagStore(conn, 384)

        results, _ = retrieve_v2(
            store=store,
            embedder=embedder,
            repo_id=repo_id,
            top_k=5,
            text="menu permessi",
        )

        self.assertIsInstance(results, list)
        if expected:
            expected_lower = expected.lower()
            self.assertTrue(
                any(expected_lower in str(item.get("source_path", "")).lower() for item in results)
            )


if __name__ == "__main__":
    unittest.main()


