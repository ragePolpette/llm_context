from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional


@dataclass(frozen=True)
class AssemblyOptions:
    max_core_files: int = 4
    max_matches_per_file: int = 3
    max_supporting_matches: int = 8
    max_entry_points: int = 6
    max_chars: int = 12000


def assemble_functional_context(
    *,
    query_text: Optional[str],
    retrieval_results: list[dict[str, Any]],
    symbol_results: Optional[list[dict[str, Any]]] = None,
    path_prefix: Optional[str] = None,
    options: Optional[AssemblyOptions] = None,
) -> dict[str, Any]:
    opts = options or AssemblyOptions()
    symbol_results = symbol_results or []
    query_terms = _extract_terms(query_text or "")

    grouped = _group_results_by_file(
        retrieval_results=retrieval_results,
        symbol_results=symbol_results,
        query_terms=query_terms,
        options=opts,
    )
    core_files = grouped[: opts.max_core_files]
    supporting_matches = _build_supporting_matches(
        retrieval_results=retrieval_results,
        core_files=core_files,
        max_items=opts.max_supporting_matches,
    )
    entry_points = _build_entry_points(
        symbol_results,
        core_paths=[str(item.get("source_path") or "") for item in core_files],
        limit=opts.max_entry_points,
    )
    assembled_context = _render_assembled_context(core_files, opts.max_chars)

    return {
        "package_version": "functional_context_v1",
        "query": {
            "text": query_text,
            "path_prefix": path_prefix,
            "terms": query_terms,
        },
        "summary": {
            "core_file_count": len(core_files),
            "supporting_match_count": len(supporting_matches),
            "symbol_hit_count": len(symbol_results),
            "query_term_count": len(query_terms),
            "assembled_char_count": len(assembled_context),
        },
        "entry_points": entry_points,
        "core_files": core_files,
        "supporting_matches": supporting_matches,
        "assembled_context": assembled_context,
    }


def _group_results_by_file(
    *,
    retrieval_results: list[dict[str, Any]],
    symbol_results: list[dict[str, Any]],
    query_terms: list[str],
    options: AssemblyOptions,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    seen_hashes_per_file: dict[str, set[str]] = {}

    for item in retrieval_results:
        source_path = str(item.get("source_path") or "").strip()
        if not source_path:
            continue
        group = groups.setdefault(
            source_path,
            {
                "source_path": source_path,
                "aggregate_score": 0.0,
                "max_score": 0.0,
                "match_count": 0,
                "symbol_hits": [],
                "matches": [],
                "query_overlap_terms": [],
                "evidence_kinds": [],
                "selection_reason": "",
            },
        )
        seen_hashes = seen_hashes_per_file.setdefault(source_path, set())
        text_hash = str(item.get("text_hash") or "")
        if text_hash and text_hash in seen_hashes:
            continue
        if text_hash:
            seen_hashes.add(text_hash)

        score = float(item.get("score", 0.0))
        group["aggregate_score"] += score
        group["max_score"] = max(group["max_score"], score)
        group["match_count"] += 1

        if len(group["matches"]) < options.max_matches_per_file:
            group["matches"].append(
                {
                    "score": score,
                    "line_start": item.get("line_start"),
                    "line_end": item.get("line_end"),
                    "chunk_index": item.get("chunk_index"),
                    "section_path": item.get("section_path") or "",
                    "snippet": (item.get("snippet") or item.get("text") or "").strip(),
                    "text": (item.get("text") or "").strip(),
                }
            )

    for symbol in symbol_results:
        source_path = str(symbol.get("source_path") or "").strip()
        if not source_path:
            continue
        group = groups.setdefault(
            source_path,
            {
                "source_path": source_path,
                "aggregate_score": 0.0,
                "max_score": 0.0,
                "match_count": 0,
                "symbol_hits": [],
                "matches": [],
                "query_overlap_terms": [],
                "evidence_kinds": [],
                "selection_reason": "",
            },
        )
        group["symbol_hits"].append(
            {
                "name": symbol.get("name"),
                "kind": symbol.get("kind"),
                "namespace": symbol.get("namespace"),
                "signature": symbol.get("signature"),
                "line_start": symbol.get("line_start"),
                "line_end": symbol.get("line_end"),
            }
        )

    for group in groups.values():
        group["symbol_hits"].sort(
            key=lambda item: (
                _symbol_kind_rank(str(item.get("kind") or "")),
                int(item.get("line_start") or 0),
            )
        )
        overlap_terms = _compute_query_overlap_terms(
            source_path=str(group.get("source_path") or ""),
            matches=group.get("matches") or [],
            symbol_hits=group.get("symbol_hits") or [],
            query_terms=query_terms,
        )
        group["query_overlap_terms"] = overlap_terms

        evidence_kinds: list[str] = []
        if group["matches"]:
            evidence_kinds.append("retrieval")
        if group["symbol_hits"]:
            evidence_kinds.append("symbol")
        if overlap_terms:
            evidence_kinds.append("query_overlap")
        group["evidence_kinds"] = evidence_kinds

        group["rank_score"] = (
            float(group["aggregate_score"])
            + float(group["max_score"]) * 0.75
            + min(len(group["symbol_hits"]), 4) * 0.15
            + min(len(overlap_terms), 4) * 0.18
            + min(int(group["match_count"] or 0), 3) * 0.05
        )
        group["selection_reason"] = _build_selection_reason(
            match_count=int(group.get("match_count") or 0),
            symbol_count=len(group.get("symbol_hits") or []),
            overlap_terms=overlap_terms,
            max_score=float(group.get("max_score") or 0.0),
        )

    return sorted(
        groups.values(),
        key=lambda item: (
            -float(item["rank_score"]),
            -len(item.get("query_overlap_terms") or []),
            -float(item["max_score"]),
            item["source_path"],
        ),
    )


def _build_supporting_matches(
    *,
    retrieval_results: list[dict[str, Any]],
    core_files: list[dict[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    core_paths = {str(item.get("source_path")) for item in core_files}
    supporting: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for item in retrieval_results:
        source_path = str(item.get("source_path") or "").strip()
        if not source_path or source_path in core_paths:
            continue
        text_hash = str(item.get("text_hash") or "")
        if text_hash and text_hash in seen_hashes:
            continue
        if text_hash:
            seen_hashes.add(text_hash)
        supporting.append(
            {
                "source_path": source_path,
                "score": float(item.get("score", 0.0)),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "snippet": (item.get("snippet") or item.get("text") or "").strip(),
            }
        )
        if len(supporting) >= max_items:
            break
    return supporting


def _build_entry_points(
    symbol_results: list[dict[str, Any]],
    *,
    core_paths: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    core_rank = {path: index for index, path in enumerate(core_paths) if path}
    ranked = sorted(
        symbol_results,
        key=lambda item: (
            core_rank.get(str(item.get("source_path") or ""), 999),
            _symbol_kind_rank(str(item.get("kind") or "")),
            str(item.get("source_path") or ""),
            int(item.get("line_start") or 0),
        ),
    )
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in ranked:
        key = (
            str(item.get("source_path") or ""),
            str(item.get("name") or ""),
            str(item.get("kind") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "source_path": item.get("source_path"),
                "name": item.get("name"),
                "kind": item.get("kind"),
                "signature": item.get("signature"),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
            }
        )
        if len(results) >= limit:
            break
    return results


def _render_assembled_context(core_files: list[dict[str, Any]], max_chars: int) -> str:
    blocks: list[str] = []
    total = 0
    for file_item in core_files:
        lines = [
            f"FILE {file_item['source_path']}",
            (
                f"aggregate_score={file_item['aggregate_score']:.4f} "
                f"max_score={file_item['max_score']:.4f} "
                f"rank_score={float(file_item.get('rank_score') or 0.0):.4f}"
            ),
        ]
        selection_reason = str(file_item.get("selection_reason") or "").strip()
        if selection_reason:
            lines.append(f"selection_reason={selection_reason}")
        overlap_terms = file_item.get("query_overlap_terms") or []
        if overlap_terms:
            lines.append("query_overlap_terms=" + ", ".join(str(item) for item in overlap_terms))
        if file_item["symbol_hits"]:
            lines.append("SYMBOLS")
            for symbol in file_item["symbol_hits"][:4]:
                name = symbol.get("name") or "(unknown)"
                kind = symbol.get("kind") or "symbol"
                line_start = symbol.get("line_start")
                line_end = symbol.get("line_end")
                signature = symbol.get("signature") or ""
                lines.append(
                    f"- {kind} {name} #L{line_start}-L{line_end} {signature}".rstrip()
                )

        lines.append("MATCHES")
        for match in file_item["matches"]:
            line_start = match.get("line_start")
            line_end = match.get("line_end")
            score = float(match.get("score", 0.0))
            text = str(match.get("text") or match.get("snippet") or "").strip()
            lines.append(f"- score={score:.4f} #L{line_start}-L{line_end}")
            lines.append(text)

        block = "\n".join(lines).strip()
        next_total = total + len(block) + (2 if blocks else 0)
        if max_chars > 0 and next_total > max_chars and blocks:
            break
        if max_chars > 0 and next_total > max_chars:
            block = block[: max_chars - total]
        blocks.append(block)
        total = total + len(block) + 2
        if max_chars > 0 and total >= max_chars:
            break
    return "\n\n".join(blocks)


def _symbol_kind_rank(kind: str) -> int:
    normalized = kind.strip().lower()
    priority = {
        "class": 0,
        "interface": 1,
        "struct": 2,
        "enum": 3,
        "type": 4,
        "method": 5,
        "function": 6,
    }
    return priority.get(normalized, 99)


def _extract_terms(text: str) -> list[str]:
    if not text:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_]{3,}", text.lower()):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _compute_query_overlap_terms(
    *,
    source_path: str,
    matches: list[dict[str, Any]],
    symbol_hits: list[dict[str, Any]],
    query_terms: list[str],
) -> list[str]:
    if not query_terms:
        return []
    haystack_parts = [source_path]
    for item in matches:
        haystack_parts.append(str(item.get("snippet") or ""))
        haystack_parts.append(str(item.get("text") or ""))
        haystack_parts.append(str(item.get("section_path") or ""))
    for item in symbol_hits:
        haystack_parts.append(str(item.get("name") or ""))
        haystack_parts.append(str(item.get("signature") or ""))
        haystack_parts.append(str(item.get("namespace") or ""))
    haystack_terms = set(_extract_terms(" ".join(haystack_parts)))
    return [term for term in query_terms if term in haystack_terms]


def _build_selection_reason(
    *,
    match_count: int,
    symbol_count: int,
    overlap_terms: list[str],
    max_score: float,
) -> str:
    reasons: list[str] = []
    if overlap_terms:
        reasons.append("query_overlap=" + ",".join(overlap_terms[:4]))
    if symbol_count:
        reasons.append(f"symbols={symbol_count}")
    if match_count:
        reasons.append(f"matches={match_count}")
    reasons.append(f"max_score={max_score:.4f}")
    return "; ".join(reasons)
