import unittest

from rag_indexer import cli
from rag_indexer.embedder import normalize_embedding_model_id
from rag_indexer.ingest import _embedder_id as ingest_embedder_id
from rag_indexer.retrieval import _embedder_id as retrieval_embedder_id


class TestCliEmbedder(unittest.TestCase):
    def test_gemini_embedder_was_removed(self):
        with self.assertRaisesRegex(ValueError, "Unknown embedder"):
            cli._build_embedder("gemini", 384)

    def test_ingest_and_query_default_to_local_st(self):
        parser = cli.build_parser()
        ingest_args = parser.parse_args(["ingest", "--dsn", "x", "--project-id", "p"])
        query_args = parser.parse_args(["query", "--dsn", "x", "--project-id", "p"])
        self.assertEqual(ingest_args.embedder, "local-st")
        self.assertEqual(query_args.embedder, "local-st")

    def test_gemini_flags_are_rejected(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["ingest", "--dsn", "x", "--project-id", "p", "--gemini-api-key", "k"])

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




def test_sentence_transformer_encode_is_serialized_across_threads(monkeypatch):
    import threading
    import time
    import types

    from rag_indexer import embedder as embedder_module

    state = {"active": 0, "max_active": 0}
    guard = threading.Lock()

    class FakeModel:
        def __init__(self, source, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with guard:
                state["active"] -= 1
            return [[0.0, 1.0] for _ in texts]

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setattr(embedder_module.importlib, "import_module", lambda name: fake_module)
    embedder = embedder_module.LocalSentenceTransformerEmbedder("fake/model")

    threads = [threading.Thread(target=embedder.embed, args=("text",)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["max_active"] == 1


if __name__ == "__main__":
    unittest.main()
