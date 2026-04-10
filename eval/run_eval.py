import json
import os
from pathlib import Path

from rag_indexer.config import load_config
from rag_indexer.db import get_connection
from rag_indexer.embedder import LocalHashEmbedder, LocalSentenceTransformerEmbedder
from rag_indexer.retrieval import retrieve_v2
from rag_indexer.store import RagStore


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    config = load_config(str(config_path))

    dsn = os.getenv("LLM_CONTEXT_DSN")
    if not dsn:
        print("LLM_CONTEXT_DSN not set.")
        return

    repo_id = os.getenv("LLM_CONTEXT_PROJECT_ID")
    if not repo_id:
        print("LLM_CONTEXT_PROJECT_ID not set.")
        return
    embedder_choice = os.getenv("LLM_CONTEXT_EVAL_EMBEDDER", "local-st").lower()
    local_model = os.getenv(
        "LLM_CONTEXT_LOCAL_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    if embedder_choice == "local-hash":
        embedder = LocalHashEmbedder(config.embedding_dim)
    else:
        embedder = LocalSentenceTransformerEmbedder(model_name=local_model)

    conn = get_connection(dsn)
    store = RagStore(conn, config.embedding_dim)

    queries_path = Path(__file__).resolve().parent / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    total = 0
    hit = 0
    mrr_sum = 0.0

    for item in queries:
        query = item["query"]
        expected = [path.lower() for path in item.get("expected_paths", [])]
        top_k = int(item.get("top_k", 8))
        results, _ = retrieve_v2(
            store=store,
            embedder=embedder,
            repo_id=repo_id,
            top_k=top_k,
            text=query,
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
            max_chunks_per_doc=config.max_chunks_per_doc,
            min_score=config.min_score,
            header_penalty=config.header_penalty,
        )
        total += 1
        rank = _first_hit_rank(results, expected)
        if rank:
            hit += 1
            mrr_sum += 1.0 / rank
        print(f"Q: {query} | rank={rank or '-'}")

    hit_at_k = hit / total if total else 0.0
    mrr = mrr_sum / total if total else 0.0
    print(f"hit@k={hit_at_k:.2f}  mrr={mrr:.2f}  n={total}")


def _first_hit_rank(results, expected_paths):
    if not expected_paths:
        return None
    for index, item in enumerate(results, start=1):
        path = str(item.get("source_path", "")).lower()
        if any(expected in path for expected in expected_paths):
            return index
    return None


if __name__ == "__main__":
    main()
