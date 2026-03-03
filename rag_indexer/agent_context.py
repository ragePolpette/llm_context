from __future__ import annotations

from typing import Any, Optional

from .db import get_connection, get_pool
from .embedder import Embedder
from .retrieval import retrieve_v2
from .store import RagStore


def build_context(
    dsn: str,
    embedder: Embedder,
    embedding_dim: int,
    project_id: str,
    query_text: Optional[str] = None,
    query_embedding: Optional[list[float]] = None,
    top_k: int = 8,
    path_prefix: Optional[str] = None,
    max_chars: int = 4000,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    max_chunks_per_doc: int = 2,
    min_score: float = 0.15,
    header_penalty: float = 0.15,
) -> tuple[str, list[dict[str, Any]]]:
    # Try to use pool first, fallback to single connection
    pool = get_pool(dsn)
    if pool is not None:
        with pool.connection() as conn:
            store = RagStore(conn, embedding_dim)
            results, _ = retrieve_v2(
                store=store,
                embedder=embedder,
                repo_id=project_id,
                top_k=top_k,
                text=query_text,
                embedding=query_embedding,
                path_prefix=path_prefix,
                doc_type=doc_type,
                language=language,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                max_chunks_per_doc=max_chunks_per_doc,
                min_score=min_score,
                header_penalty=header_penalty,
            )
    else:
        # Fallback if psycopg_pool not available
        conn = get_connection(dsn)
        try:
            store = RagStore(conn, embedding_dim)
            results, _ = retrieve_v2(
                store=store,
                embedder=embedder,
                repo_id=project_id,
                top_k=top_k,
                text=query_text,
                embedding=query_embedding,
                path_prefix=path_prefix,
                doc_type=doc_type,
                language=language,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                max_chunks_per_doc=max_chunks_per_doc,
                min_score=min_score,
                header_penalty=header_penalty,
            )
        finally:
            conn.close()

    context = format_context(results, max_chars=max_chars)
    return context, results


def format_context(results: list[dict[str, Any]], max_chars: int = 4000) -> str:
    blocks: list[str] = []
    total = 0
    for item in results:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        source_path = item.get("source_path", "")
        chunk_index = item.get("chunk_index", "")
        score = item.get("score", item.get("score_final", 0.0))
        line_start = item.get("line_start")
        line_end = item.get("line_end")
        if line_start and line_end:
            header = f"[{source_path}#L{line_start}-L{line_end}] score={score:.4f}"
        else:
            header = f"[{source_path}]#chunk={chunk_index} score={score:.4f}"
        block = header + "\n" + text
        next_total = total + len(block)
        if next_total > max_chars and blocks:
            break
        if next_total > max_chars:
            block = block[:max_chars]
        blocks.append(block)
        total = total + len(block) + 2
        if total >= max_chars:
            break
    return "\n\n".join(blocks)
