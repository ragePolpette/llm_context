"""
Context snapshot exporter.

Produces static pre-computed files that an agent can load as initial context
without needing live DB queries.  Run via:
  python cli.py export-snapshot --project-id <id> --output <dir>

Output layout under <output>/<project_id>/:
  snapshot_index.json     — overview + stats
  symbols_catalog.json    — all symbols with path, line, kind, callers
  functional_areas.json   — areas from scope_map with representative files/symbols
  top_files.json          — top-N files by aggregated relevance
  CONTEXT.md              — human/agent readable orientation document
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import AppConfig
from .context_assembler import assemble_functional_context
from .db import get_connection
from .embedder import Embedder
from .project_registry import ProjectRecord
from .retrieval import retrieve_v2
from .store import RagStore

log = logging.getLogger(__name__)


@dataclass
class SnapshotResult:
    project_id: str
    output_dir: Path
    symbol_count: int = 0
    top_file_count: int = 0
    area_count: int = 0
    generated_at: str = ""
    files_written: list[str] = field(default_factory=list)


def export_snapshot(
    *,
    dsn: str,
    embedder: Embedder,
    embedding_dim: int,
    project_id: str,
    project_record: ProjectRecord,
    output_dir: Path,
    config: AppConfig,
    top_files_limit: int = 20,
) -> SnapshotResult:
    """
    Export a pre-computed context snapshot for a project.

    Connects to the DB once, runs retrieval for each functional area, collects
    symbols + caller sites, and writes static JSON + CONTEXT.md files.
    """
    result = SnapshotResult(
        project_id=project_id,
        output_dir=output_dir,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(dsn)
    try:
        store = RagStore(conn, embedding_dim)
        has_refs = store.has_symbol_refs_table()

        repo_stats = store.get_repo_stats(project_id)
        symbols_raw = _query_all_symbols(store, project_id)
        result.symbol_count = len(symbols_raw)

        # --- Functional areas via scope_map ---
        areas = _build_functional_areas(
            store=store,
            embedder=embedder,
            config=config,
            project_id=project_id,
            symbols_raw=symbols_raw,
        )
        result.area_count = len(areas)

        # --- Top files by aggregated score across all areas ---
        top_files = _compute_top_files(areas, limit=top_files_limit)
        result.top_file_count = len(top_files)

        # --- Enrich symbols with caller sites (top symbols only) ---
        top_symbol_names = {
            sym["name"]
            for f in top_files
            for sym in f.get("key_symbols", [])
        }
        symbol_callers: dict[str, list[dict[str, Any]]] = {}
        if has_refs:
            for name in top_symbol_names:
                callers = store.query_caller_sites(name, project_id, limit=20)
                if callers:
                    symbol_callers[name] = callers
    finally:
        conn.close()

    # --- Build output data structures ---
    snapshot_index = _build_snapshot_index(
        project_id=project_id,
        project_record=project_record,
        repo_stats=repo_stats,
        generated_at=result.generated_at,
        area_count=result.area_count,
        top_file_count=result.top_file_count,
        symbol_count=result.symbol_count,
        has_caller_sites=bool(symbol_callers),
    )

    symbols_catalog = _build_symbols_catalog(symbols_raw, symbol_callers)
    functional_areas_out = _build_areas_output(areas, symbol_callers)
    top_files_out = _build_top_files_output(top_files, symbol_callers)

    # --- Write JSON files ---
    files = {
        "snapshot_index.json": snapshot_index,
        "symbols_catalog.json": symbols_catalog,
        "functional_areas.json": functional_areas_out,
        "top_files.json": top_files_out,
    }
    for filename, data in files.items():
        path = output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        result.files_written.append(str(path))
        log.info("Wrote %s", path)

    # --- Write CONTEXT.md ---
    context_md_path = output_dir / "CONTEXT.md"
    context_md = _render_context_md(
        project_id=project_id,
        project_record=project_record,
        snapshot_index=snapshot_index,
        areas=functional_areas_out,
        top_files=top_files_out,
        symbol_callers=symbol_callers,
    )
    context_md_path.write_text(context_md, encoding="utf-8")
    result.files_written.append(str(context_md_path))
    log.info("Wrote %s", context_md_path)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query_all_symbols(store: RagStore, project_id: str) -> list[dict[str, Any]]:
    """Fetch all symbols for the project from the DB."""
    sql_text = (
        "SELECT s.name, s.kind, s.namespace, s.line_start, s.line_end, s.signature, "
        "s.language, d.path "
        "FROM symbols s "
        "JOIN documents d ON d.doc_id = s.doc_id "
        "WHERE s.repo_id = @Valore0 AND d.deleted_at IS NULL "
        "ORDER BY d.path, s.line_start"
    )
    from .db import execute_params
    with store.conn.cursor() as cur:
        execute_params(cur, sql_text, [project_id])
        rows = cur.fetchall()
    return [
        {
            "name": row[0],
            "kind": row[1],
            "namespace": row[2],
            "line_start": row[3],
            "line_end": row[4],
            "signature": row[5],
            "language": row[6],
            "source_path": row[7],
        }
        for row in rows
    ]


def _build_functional_areas(
    *,
    store: RagStore,
    embedder: Embedder,
    config: AppConfig,
    project_id: str,
    symbols_raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    For each entry in scope_map, run retrieval with area keywords as query
    and assemble functional context.  Returns list of area dicts.
    """
    areas: list[dict[str, Any]] = []
    scope_map = config.scope_map or {}

    if not scope_map:
        # Fallback: run a single generic query to capture top files
        log.info("No scope_map defined; running generic top-files query for project=%s", project_id)
        results, _ = retrieve_v2(
            store=store,
            embedder=embedder,
            repo_id=project_id,
            top_k=20,
            text="main functionality entry point",
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
            max_chunks_per_doc=3,
            min_score=config.min_score,
            header_penalty=config.header_penalty,
        )
        fc = assemble_functional_context(
            query_text="main functionality entry point",
            retrieval_results=results,
            symbol_results=[],
        )
        areas.append({
            "area_name": "general",
            "path_prefix": None,
            "query_terms": [],
            "core_files": fc.get("core_files", []),
            "supporting_matches": fc.get("supporting_matches", []),
            "entry_points": fc.get("entry_points", []),
        })
        return areas

    for path_prefix, query_terms in scope_map.items():
        if not query_terms:
            continue
        query_text = " ".join(query_terms[:3])
        try:
            results, _ = retrieve_v2(
                store=store,
                embedder=embedder,
                repo_id=project_id,
                top_k=12,
                text=query_text,
                path_prefix=path_prefix,
                vector_weight=config.vector_weight,
                keyword_weight=config.keyword_weight,
                max_chunks_per_doc=3,
                min_score=config.min_score,
                header_penalty=config.header_penalty,
            )
        except Exception as exc:
            log.warning("Retrieval failed for area %r: %s", path_prefix, exc)
            results = []

        # Collect symbols from files that appeared in results
        result_paths = {str(r.get("source_path") or "") for r in results}
        area_symbols = [s for s in symbols_raw if s.get("source_path") in result_paths]

        fc = assemble_functional_context(
            query_text=query_text,
            retrieval_results=results,
            symbol_results=area_symbols,
        )
        areas.append({
            "area_name": _path_to_area_name(path_prefix),
            "path_prefix": path_prefix,
            "query_terms": list(query_terms),
            "core_files": fc.get("core_files", []),
            "supporting_matches": fc.get("supporting_matches", []),
            "entry_points": fc.get("entry_points", []),
        })
        log.info(
            "Area %r: %d core_files, %d entry_points",
            path_prefix,
            len(fc.get("core_files", [])),
            len(fc.get("entry_points", [])),
        )

    return areas


def _path_to_area_name(path_prefix: str) -> str:
    """Convert a path prefix like 'pubblico\\api\\' to a readable area name."""
    parts = path_prefix.replace("\\", "/").strip("/").split("/")
    return "/".join(p for p in parts if p)


def _compute_top_files(areas: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Aggregate file scores across all areas and return top-N."""
    file_scores: dict[str, dict[str, Any]] = {}
    for area in areas:
        for cf in area.get("core_files", []):
            path = str(cf.get("source_path") or "").strip()
            if not path:
                continue
            if path not in file_scores:
                file_scores[path] = {
                    "source_path": path,
                    "aggregate_score": 0.0,
                    "area_count": 0,
                    "key_symbols": [],
                    "functional_role": cf.get("functional_role", "supporting"),
                }
            entry = file_scores[path]
            entry["aggregate_score"] += float(cf.get("aggregate_score", 0.0))
            entry["area_count"] += 1
            # Merge symbols from all areas
            existing_syms = {
                (s["name"], s.get("kind")): True for s in entry["key_symbols"]
            }
            for sym in cf.get("symbol_hits", []):
                key = (sym.get("name"), sym.get("kind"))
                if key not in existing_syms:
                    entry["key_symbols"].append(sym)
                    existing_syms[key] = True

    sorted_files = sorted(
        file_scores.values(),
        key=lambda x: (-x["aggregate_score"], -x["area_count"]),
    )
    return sorted_files[:limit]


def _build_snapshot_index(
    *,
    project_id: str,
    project_record: ProjectRecord,
    repo_stats: dict[str, int],
    generated_at: str,
    area_count: int,
    top_file_count: int,
    symbol_count: int,
    has_caller_sites: bool,
) -> dict[str, Any]:
    return {
        "snapshot_version": "v1",
        "project_id": project_id,
        "display_name": project_record.display_name,
        "generated_at": generated_at,
        "indexed_documents": repo_stats.get("indexed_documents", 0),
        "indexed_chunks": repo_stats.get("indexed_chunks", 0),
        "indexed_symbols": repo_stats.get("indexed_symbols", 0),
        "functional_areas": area_count,
        "top_files": top_file_count,
        "symbol_count": symbol_count,
        "caller_sites_available": has_caller_sites,
        "usage_note": (
            "Load CONTEXT.md as initial context before making live RAG queries. "
            "Use functional_areas.json to understand project structure. "
            "Use symbols_catalog.json for precise symbol lookup."
        ),
    }


def _build_symbols_catalog(
    symbols_raw: list[dict[str, Any]],
    symbol_callers: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    catalog = []
    for sym in symbols_raw:
        entry: dict[str, Any] = {
            "name": sym["name"],
            "kind": sym["kind"],
            "source_path": sym["source_path"],
            "line_start": sym["line_start"],
            "line_end": sym["line_end"],
            "signature": sym.get("signature"),
            "namespace": sym.get("namespace"),
            "language": sym.get("language"),
        }
        callers = symbol_callers.get(sym["name"])
        if callers:
            entry["referenced_in"] = callers
            entry["reference_count"] = len(callers)
        catalog.append(entry)
    return catalog


def _build_areas_output(
    areas: list[dict[str, Any]],
    symbol_callers: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for area in areas:
        core_files_out = []
        for cf in area.get("core_files", []):
            file_entry: dict[str, Any] = {
                "source_path": cf.get("source_path"),
                "functional_role": cf.get("functional_role"),
                "aggregate_score": cf.get("aggregate_score"),
                "match_count": cf.get("match_count"),
            }
            key_syms = []
            for sym in cf.get("symbol_hits", [])[:6]:
                sym_entry: dict[str, Any] = {
                    "name": sym.get("name"),
                    "kind": sym.get("kind"),
                    "line_start": sym.get("line_start"),
                    "signature": sym.get("signature"),
                }
                callers = symbol_callers.get(str(sym.get("name") or ""))
                if callers:
                    sym_entry["reference_count"] = len(callers)
                key_syms.append(sym_entry)
            if key_syms:
                file_entry["key_symbols"] = key_syms
            core_files_out.append(file_entry)

        output.append({
            "area_name": area["area_name"],
            "path_prefix": area.get("path_prefix"),
            "query_terms": area.get("query_terms", []),
            "core_files": core_files_out,
            "entry_points": [
                {
                    "name": ep.get("name"),
                    "kind": ep.get("kind"),
                    "source_path": ep.get("source_path"),
                    "line_start": ep.get("line_start"),
                    "signature": ep.get("signature"),
                }
                for ep in area.get("entry_points", [])[:6]
            ],
        })
    return output


def _build_top_files_output(
    top_files: list[dict[str, Any]],
    symbol_callers: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for rank, tf in enumerate(top_files, start=1):
        entry: dict[str, Any] = {
            "rank": rank,
            "source_path": tf["source_path"],
            "aggregate_score": round(tf["aggregate_score"], 4),
            "area_count": tf["area_count"],
            "functional_role": tf.get("functional_role", "supporting"),
        }
        key_syms = []
        for sym in tf.get("key_symbols", [])[:8]:
            sym_entry: dict[str, Any] = {
                "name": sym.get("name"),
                "kind": sym.get("kind"),
                "line_start": sym.get("line_start"),
            }
            callers = symbol_callers.get(str(sym.get("name") or ""))
            if callers:
                sym_entry["reference_count"] = len(callers)
            key_syms.append(sym_entry)
        if key_syms:
            entry["key_symbols"] = key_syms
        output.append(entry)
    return output


def _render_context_md(
    *,
    project_id: str,
    project_record: ProjectRecord,
    snapshot_index: dict[str, Any],
    areas: list[dict[str, Any]],
    top_files: list[dict[str, Any]],
    symbol_callers: dict[str, list[dict[str, Any]]],
) -> str:
    lines: list[str] = []

    lines.append(f"# Project Context: {project_record.display_name}")
    lines.append(f"\nProject ID: `{project_id}`  ")
    lines.append(f"Generated: {snapshot_index.get('generated_at', '')}  ")
    lines.append(
        f"Index: {snapshot_index.get('indexed_documents', 0)} docs · "
        f"{snapshot_index.get('indexed_chunks', 0)} chunks · "
        f"{snapshot_index.get('indexed_symbols', 0)} symbols"
    )
    lines.append(
        "\n> **Usage:** Load this file as initial context before querying the live RAG system. "
        "It provides a pre-compiled map of the codebase so you can orient yourself without "
        "spending query budget on exploration."
    )

    # --- Functional Areas ---
    if areas:
        lines.append("\n## Functional Areas\n")
        for area in areas:
            area_name = area.get("area_name") or "unknown"
            path_prefix = area.get("path_prefix", "")
            query_terms = area.get("query_terms", [])
            lines.append(f"### {area_name}")
            if path_prefix:
                lines.append(f"Path prefix: `{path_prefix}`  ")
            if query_terms:
                lines.append(f"Keywords: {', '.join(query_terms[:5])}")
            lines.append("")

            core_files = area.get("core_files", [])
            if core_files:
                lines.append("**Key files:**")
                for cf in core_files[:5]:
                    path = cf.get("source_path", "")
                    role = cf.get("functional_role", "")
                    score = cf.get("aggregate_score", 0.0)
                    lines.append(f"- `{path}` [{role}, score={score:.2f}]")
                    for sym in cf.get("key_symbols", [])[:3]:
                        name = sym.get("name", "")
                        kind = sym.get("kind", "")
                        line = sym.get("line_start", "")
                        sig = sym.get("signature", "")
                        ref_count = sym.get("reference_count", 0)
                        ref_note = f" — {ref_count} caller(s)" if ref_count else ""
                        lines.append(f"  - `{name}` ({kind}, line {line}){ref_note}")
                        if sig:
                            lines.append(f"    `{sig[:100]}`")

            entry_points = area.get("entry_points", [])
            if entry_points:
                lines.append("\n**Entry points:**")
                for ep in entry_points[:4]:
                    name = ep.get("name", "")
                    kind = ep.get("kind", "")
                    path = ep.get("source_path", "")
                    line = ep.get("line_start", "")
                    sig = ep.get("signature", "")
                    lines.append(f"- `{name}` ({kind}) — `{path}` line {line}")
                    if sig:
                        lines.append(f"  `{sig[:100]}`")
            lines.append("")

    # --- Top Files ---
    if top_files:
        lines.append("## Most Relevant Files\n")
        lines.append("Files ranked by aggregate relevance score across all functional areas:\n")
        for tf in top_files[:15]:
            rank = tf.get("rank", "")
            path = tf.get("source_path", "")
            score = tf.get("aggregate_score", 0.0)
            role = tf.get("functional_role", "")
            area_count = tf.get("area_count", 0)
            lines.append(f"{rank}. `{path}`  ")
            lines.append(
                f"   score={score:.2f} · {area_count} area(s) · role={role}"
            )
            syms = tf.get("key_symbols", [])
            if syms:
                sym_list = ", ".join(
                    f"`{s.get('name')}`({s.get('kind')})" for s in syms[:4]
                )
                lines.append(f"   symbols: {sym_list}")

    # --- Symbol Callers Summary ---
    if symbol_callers:
        lines.append("\n## Impact Map (caller sites)\n")
        lines.append(
            "Symbols with known call sites — useful for impact analysis "
            "when planning a code change:\n"
        )
        sorted_callers = sorted(
            symbol_callers.items(), key=lambda kv: -len(kv[1])
        )
        for name, callers in sorted_callers[:20]:
            lines.append(f"### `{name}` — {len(callers)} reference(s)")
            for c in callers[:5]:
                lines.append(f"- `{c['path']}` line {c['line']}")
            if len(callers) > 5:
                lines.append(f"- _(+{len(callers) - 5} more)_")
            lines.append("")

    lines.append("\n---")
    lines.append(
        "_Snapshot generated by llm-context. "
        "Use `rag_context` / `rag_full_context` for live queries on specific topics._"
    )

    return "\n".join(lines) + "\n"
