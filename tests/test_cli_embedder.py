import unittest

import cli
from rag_indexer.embedder import normalize_embedding_model_id
from rag_indexer.ingest import _embedder_id as ingest_embedder_id
from rag_indexer.retrieval import _embedder_id as retrieval_embedder_id


class TestCliEmbedder(unittest.TestCase):
    def test_gemini_embedder_fails_cleanly_without_name_error(self):
        with self.assertRaises(RuntimeError):
            cli._build_embedder("gemini", 384)

    def test_normalize_huggingface_snapshot_model_id(self):
        snapshot_path = (
            r"C:\Users\Gianmarco\.cache\huggingface\hub"
            r"\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
            r"\snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
        )

        self.assertEqual(
            normalize_embedding_model_id(snapshot_path),
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

    def test_retrieval_and_ingest_use_stable_model_id(self):
        class FakeEmbedder:
            model_name = (
                "/home/user/.cache/huggingface/hub/"
                "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/"
                "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
            )

        self.assertEqual(
            retrieval_embedder_id(FakeEmbedder()),
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.assertEqual(
            ingest_embedder_id(FakeEmbedder()),
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )


if __name__ == "__main__":
    unittest.main()


