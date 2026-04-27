from __future__ import annotations

import re
from typing import Any, Optional

from .embedder import Embedder, normalize_embedding_model_id
from .store import RagStore


def retrieve_v2(
    store: RagStore,
    embedder: Embedder,
    repo_id: str,
    top_k: int,
    text: Optional[str] = None,
    embedding: Optional[list[float]] = None,
    path_prefix: Optional[str] = None,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    max_chunks_per_doc: int = 2,
    min_score: float = 0.15,
    header_penalty: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if embedding is None:
        if text is None:
            raise ValueError("text or embedding is required")
        embedding = embedder.embed(text)

    model_id = _embedder_id(embedder)
    candidate_k = max(top_k * 4, top_k)
    if path_prefix:
        path_prefix = path_prefix.replace("/", "\\").lower()

    vector_hits = store.query_vector(
        embedding=embedding,
        repo_id=repo_id,
        model_id=model_id,
        top_k=candidate_k,
        path_prefix=path_prefix,
        doc_type=doc_type,
        language=language,
    )

    keyword_hits: list[dict[str, Any]] = []
    if text:
        keyword_hits = store.query_keyword(
            query_text=text,
            repo_id=repo_id,
            top_k=candidate_k,
            path_prefix=path_prefix,
            doc_type=doc_type,
            language=language,
        )

    max_rank = max((hit.get("rank", 0.0) for hit in keyword_hits), default=0.0)
    keyword_norm = max_rank if max_rank > 0 else 1.0

    merged: dict[Any, dict[str, Any]] = {}
    for hit in vector_hits:
        score_vector = max(0.0, 1.0 - float(hit.get("distance", 1.0)))
        merged[hit["chunk_id"]] = {
            **hit,
            "score_vector": score_vector,
            "score_keyword": 0.0,
        }

    for hit in keyword_hits:
        score_keyword = float(hit.get("rank", 0.0)) / keyword_norm
        existing = merged.get(hit["chunk_id"])
        if existing:
            existing["score_keyword"] = max(existing.get("score_keyword", 0.0), score_keyword)
        else:
            merged[hit["chunk_id"]] = {
                **hit,
                "score_vector": 0.0,
                "score_keyword": score_keyword,
            }

    results = []
    for item in merged.values():
        score = vector_weight * float(item.get("score_vector", 0.0)) + keyword_weight * float(
            item.get("score_keyword", 0.0)
        )
        if item.get("is_header"):
            score = max(0.0, score - header_penalty)
        if score < min_score:
            continue
        item["score"] = score
        item["snippet"] = _make_snippet(item.get("text", ""), query_text=text)
        results.append(item)

    results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    deduped: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    per_doc: dict[Any, int] = {}
    for item in results:
        text_hash = str(item.get("text_hash", ""))
        if text_hash and text_hash in seen_hashes:
            continue
        doc_id = item.get("doc_id")
        if doc_id is not None:
            count = per_doc.get(doc_id, 0)
            if count >= max_chunks_per_doc:
                continue
            per_doc[doc_id] = count + 1
        if text_hash:
            seen_hashes.add(text_hash)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    explain = {
        "repo_id": repo_id,
        "model_id": model_id,
        "top_k": top_k,
        "path_prefix": path_prefix,
        "doc_type": doc_type,
        "language": language,
        "vector_weight": vector_weight,
        "keyword_weight": keyword_weight,
        "max_chunks_per_doc": max_chunks_per_doc,
        "min_score": min_score,
    }
    return deduped, explain


def _make_snippet(
    text: str,
    max_len: int = 300,
    query_text: Optional[str] = None,
) -> str:
    if len(text) <= max_len:
        return text
    if query_text:
        terms = sorted(
            {
                token.lower()
                for token in re.findall(r"\w+", query_text)
                if len(token) >= 3
            },
            key=len,
            reverse=True,
        )
        lower_text = text.lower()
        for term in terms:
            index = lower_text.find(term)
            if index < 0:
                continue
            half = max(1, max_len // 2)
            start = max(0, index - half)
            end = min(len(text), start + max_len)
            start = max(0, end - max_len)
            body = text[start:end].strip()
            if not body:
                break
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            return prefix + body + suffix
    return text[: max_len - 3] + "..."


def _embedder_id(embedder: Embedder) -> str:
    model_name = getattr(embedder, "model_name", None)
    if model_name:
        return normalize_embedding_model_id(model_name)
    return embedder.__class__.__name__.lower()
