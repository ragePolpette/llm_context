"""
MCP Handler for RAG context/search tools.

Shared logic for both stdio and HTTP MCP servers.
"""

from __future__ import annotations

import json
import hashlib
import ipaddress
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Optional
from urllib.parse import urlsplit

from rag_indexer import __version__
from rag_indexer.agent_context import build_context
from rag_indexer.config import load_config
from rag_indexer.context_assembler import assemble_functional_context
from rag_indexer.db import get_connection, get_pool, inspect_database_runtime
from rag_indexer.embedder import (
    DummyEmbedder,
    Embedder,
    LocalHashEmbedder,
    LocalSentenceTransformerEmbedder,
)
from rag_indexer.path_utils import parse_bool, resolve_path_prefix
from rag_indexer.project_registry import load_project_registry
from rag_indexer.store import RagStore
from rag_indexer.work_item_mapper import map_work_item_to_codebase


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles UUID objects."""

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


class MCPHandler:
    """Handles MCP tool calls (shared logic between stdio and HTTP servers)."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._project_root = Path(__file__).resolve().parent.parent
        self._configure_local_model_dirs()
        if config_path is None:
            config_path = str(self._project_root / "config.yaml")
        self._config_path = str(Path(config_path).resolve())
        self._config = load_config(config_path)
        self._project_registry = load_project_registry(
            self._config.projects_registry_path,
            self._config.projects_state_path,
        )
        self._registry_lock = threading.Lock()
        self._registry_signature_value = self._registry_signature()
        self._embedder_lock = threading.Lock()
        self._default_dsn = self._require_default_dsn()
        self._default_embedder = os.getenv("LLM_CONTEXT_EMBEDDER", "local-st")
        self._allow_embedder_fallback = parse_bool(
            os.getenv("LLM_CONTEXT_ALLOW_EMBEDDER_FALLBACK", "true")
        )
        self._fallback_embedder = str(
            os.getenv("LLM_CONTEXT_FALLBACK_EMBEDDER", "local-hash")
        ).strip().lower()
        self._default_local_model = os.getenv(
            "LLM_CONTEXT_LOCAL_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self._default_embedding_dim = int(
            os.getenv("LLM_CONTEXT_EMBEDDING_DIM", str(self._config.embedding_dim))
        )
        self._runtime_name = str(
            os.getenv("LLM_CONTEXT_RUNTIME_NAME", "default")
        ).strip() or "default"
        self._store_target_name = str(
            os.getenv("LLM_CONTEXT_STORE_TARGET", "")
        ).strip() or None
        self._max_query_embedding_items = int(
            os.getenv("LLM_CONTEXT_MAX_QUERY_EMBEDDING_ITEMS", "4096")
        )
        self._default_max_chars = int(os.getenv("LLM_CONTEXT_MAX_CHARS", "12000"))
        self._max_top_k = max(1, int(os.getenv("LLM_CONTEXT_MAX_TOP_K", "50")))
        self._max_symbol_limit = max(1, int(os.getenv("LLM_CONTEXT_MAX_SYMBOL_LIMIT", "200")))
        self._max_chars_limit = max(1, int(os.getenv("LLM_CONTEXT_MAX_CHARS_LIMIT", "100000")))
        self._max_query_chars = max(1, int(os.getenv("LLM_CONTEXT_MAX_QUERY_CHARS", "8000")))
        self._embedder_cache: dict[str, Embedder] = {}
        self._query_cache: dict[str, dict[str, Any]] = {}
        self._query_cache_lock = threading.Lock()
        self._query_cache_max = int(os.getenv("LLM_CONTEXT_CACHE_MAX", "500"))
        self._query_cache_ttl = int(os.getenv("LLM_CONTEXT_CACHE_TTL_SECONDS", "1800"))
        self._db_probe_cache: Optional[dict[str, Any]] = None
        self._db_probe_cache_at = 0.0
        self._db_probe_lock = threading.Lock()
        self._db_probe_ttl_seconds = max(
            0,
            int(os.getenv("LLM_CONTEXT_DB_PROBE_TTL_SECONDS", "5")),
        )
        self._db_connect_timeout_seconds = max(
            1,
            int(os.getenv("LLM_CONTEXT_DB_CONNECT_TIMEOUT_SECONDS", "3")),
        )
        self._ready = threading.Event()

        # Start warmup in background
        sys.stderr.write("[INFO] Starting embedder warmup...\n")
        threading.Thread(target=self._warmup_embedder, daemon=True).start()

    def _require_default_dsn(self) -> str:
        dsn = str(os.getenv("LLM_CONTEXT_DSN", "")).strip()
        if not dsn:
            raise RuntimeError("LLM_CONTEXT_DSN is required for llm-context MCP")
        return dsn

    def _configure_local_model_dirs(self) -> None:
        """Force local model/cache directories under the MCP root by default."""
        default_models_dir = self._project_root / ".local" / "models"
        models_dir = Path(os.getenv("MCP_MODELS_DIR", str(default_models_dir))).expanduser()
        if not models_dir.is_absolute():
            models_dir = (self._project_root / models_dir).resolve()

        hf_home = Path(
            os.getenv("HF_HOME", str(models_dir / "huggingface"))
        ).expanduser()
        if not hf_home.is_absolute():
            hf_home = (self._project_root / hf_home).resolve()

        os.environ.setdefault("MCP_MODELS_DIR", str(models_dir))
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
        os.environ.setdefault(
            "SENTENCE_TRANSFORMERS_HOME",
            str(hf_home / "sentence_transformers"),
        )

    def _warmup_embedder(self) -> None:
        try:
            self._get_embedder()
            sys.stderr.write(
                f"[INFO] Embedder warmup completed: {self._default_local_model}\n"
            )
        except Exception as e:
            sys.stderr.write(f"[WARN] Embedder warmup failed: {e}\n")
        finally:
            self._ready.set()

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Wait for the embedder to be ready."""
        return self._ready.wait(timeout=timeout)

    def handle_message(self, message: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Handle an MCP JSON-RPC message.

        Args:
            message: The incoming JSON-RPC message

        Returns:
            JSON-RPC response or None for notifications
        """
        if not isinstance(message, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }

        method = message.get("method")
        msg_id = message.get("id")
        sys.stderr.write(f"[DEBUG] handle_message: method='{method}', id={msg_id}\n")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {},
                        "prompts": {},
                    },
                    "serverInfo": {"name": "llm-context-mcp", "version": __version__},
                },
            }

        if method == "initialized":
            return None

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        tool_rag_context(),
                        tool_rag_search(),
                        tool_rag_full_context(),
                        tool_map_work_item_to_codebase(),
                        tool_context_info(),
                        tool_symbol_search(),
                        tool_list_projects(),
                        tool_get_project_info(),
                        tool_get_project_snapshot(),
                    ]
                },
            }

        # Capability endpoints often requested by newer MCP clients
        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"resources": []},
            }

        if method == "resources/templates/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"resourceTemplates": []},
            }

        if method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"prompts": []},
            }

        if method == "tools/call":
            params = message.get("params", {}) or {}
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            try:
                if name == "rag_context":
                    payload = self._run_rag_context(args)
                elif name == "rag_search":
                    payload = self._run_rag_search(args)
                elif name == "rag_full_context":
                    payload = self._run_rag_full_context(args)
                elif name == "map_work_item_to_codebase":
                    payload = self._run_map_work_item_to_codebase(args)
                elif name == "context_info":
                    payload = self._run_context_info()
                elif name == "symbol_search":
                    payload = self._run_symbol_search(args)
                elif name == "list_projects":
                    payload = self._run_list_projects()
                elif name == "get_project_info":
                    payload = self._run_get_project_info(args)
                elif name == "get_project_snapshot":
                    payload = self._run_get_project_snapshot(args)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32000,
                        "message": str(exc),
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": format_tool_text(name, args, payload),
                        }
                    ]
                },
            }

        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": msg_id, "result": None}

        if msg_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _preview_value(self, value: Any, *, limit: int = 240) -> Any:
        if isinstance(value, str):
            compact = " ".join(value.split())
            return compact[:limit]
        return value

    def _log_activity(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": "llm-context-mcp",
            "event": event,
            **payload,
        }
        sys.stderr.write(f"[MCP_ACTIVITY] {json.dumps(record, ensure_ascii=True, cls=UUIDEncoder)}\n")

    def _make_cache_key(self, params: dict[str, Any]) -> Optional[str]:
        """Build a deterministic cache key from resolved query params. Returns None if not cacheable."""
        query_text = params.get("query_text")
        if not query_text:
            return None
        key_data = {
            "project_id": str(params.get("project_id") or ""),
            "query_text": str(query_text).strip().lower(),
            "path_prefix": str(params.get("path_prefix") or "").strip().lower(),
            "file": str(params.get("file") or "").strip().lower(),
            "top_k": int(params.get("top_k") or 0),
            "max_chars": int(params.get("max_chars") or 0),
            "auto_scope": bool(params.get("auto_scope")),
            "doc_type": str(params.get("doc_type") or ""),
            "language": str(params.get("language") or ""),
            "embedder": str(self._default_embedder).strip().lower(),
            "local_model": str(self._default_local_model),
            "embedding_dim": int(self._default_embedding_dim),
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[dict[str, Any]]:
        with self._query_cache_lock:
            entry = self._query_cache.get(cache_key)
            if entry is None:
                return None
            if monotonic() - entry["ts"] > self._query_cache_ttl:
                del self._query_cache[cache_key]
                return None
            return entry["payload"]

    def _set_cache(self, cache_key: str, payload: dict[str, Any]) -> None:
        with self._query_cache_lock:
            if len(self._query_cache) >= self._query_cache_max:
                oldest = min(self._query_cache, key=lambda k: self._query_cache[k]["ts"])
                del self._query_cache[oldest]
            self._query_cache[cache_key] = {"payload": payload, "ts": monotonic()}

    def _run_rag_context(self, args: dict[str, Any], *, tool_name: str = "rag_context") -> dict[str, Any]:
        """Execute rag_context tool."""
        query_text = args.get("query_text")
        query_embedding = args.get("query_embedding")
        if query_text is None and query_embedding is None:
            raise ValueError("query_text or query_embedding is required")
        if query_text is not None and len(str(query_text)) > self._max_query_chars:
            raise ValueError(
                f"query_text exceeds maximum allowed length ({self._max_query_chars} characters)"
            )
        self._reject_client_embedder_overrides(args)

        project_id = self._resolve_project_id(args, tool_name=tool_name)
        top_k = self._top_k_arg(args)
        path_prefix = args.get("path_prefix")
        file_path = args.get("file")
        auto_scope = parse_bool(str(args.get("auto_scope", True)))
        explicit_scope = bool(path_prefix or file_path)
        max_chars = self._max_chars_arg(args)
        doc_type = args.get("doc_type") or self._config.default_doc_type
        language = args.get("language")
        embedding_dim = self._default_embedding_dim
        self._validate_query_embedding(query_embedding, embedding_dim)

        cache_key: Optional[str] = None
        if query_embedding is None and tool_name == "rag_context":
            cache_key = self._make_cache_key(
                {
                    "project_id": project_id,
                    "query_text": query_text,
                    "path_prefix": path_prefix,
                    "file": file_path,
                    "top_k": top_k,
                    "max_chars": max_chars,
                    "auto_scope": auto_scope,
                    "doc_type": doc_type,
                    "language": language,
                }
            )
            if cache_key:
                cached = self._get_cached(cache_key)
                if cached is not None:
                    self._log_activity("query_cache_hit", {"tool": tool_name, "project_id": project_id})
                    return cached

        embedder = self._get_embedder()

        project_record = self._project_registry.get_project(project_id)
        resolved_prefix = resolve_path_prefix(
            path_prefix=path_prefix,
            file_path=file_path,
            query_text=query_text,
            auto_scope=auto_scope,
            scope_map=self._project_setting(project_record, "scope_map"),
            path_aliases=self._project_setting(project_record, "path_aliases"),
        )
        self._log_activity(
            "query_in",
            {
                "tool": tool_name,
                "project_id": project_id,
                "query_text": self._preview_value(query_text),
                "query_embedding_provided": query_embedding is not None,
                "top_k": top_k,
                "path_prefix": resolved_prefix,
                "file": file_path,
                "doc_type": doc_type,
                "language": language,
            },
        )

        def run_context_search(*, path_prefix: Optional[str], min_score: float):
            return build_context(
                dsn=self._default_dsn,
                embedder=embedder,
                embedding_dim=embedding_dim,
                project_id=project_id,
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                path_prefix=path_prefix,
                max_chars=max_chars,
                doc_type=doc_type,
                language=language,
                vector_weight=self._config.vector_weight,
                keyword_weight=self._config.keyword_weight,
                max_chunks_per_doc=self._config.max_chunks_per_doc,
                min_score=min_score,
                header_penalty=self._config.header_penalty,
                fts_config=self._config.fts_config,
            )

        effective_min_score = self._config.min_score
        context, results = run_context_search(
            path_prefix=resolved_prefix,
            min_score=effective_min_score,
        )
        auto_scope_fallback_used = False
        relaxed_min_score_fallback_used = False
        initial_resolved_prefix = resolved_prefix
        if not results and resolved_prefix and auto_scope and not explicit_scope:
            context, results = run_context_search(
                path_prefix=None,
                min_score=effective_min_score,
            )
            resolved_prefix = None
            auto_scope_fallback_used = True
        min_score_floor = min(self._config.min_score, self._config.min_score_floor)
        if not results and effective_min_score > min_score_floor:
            context, results = run_context_search(
                path_prefix=resolved_prefix,
                min_score=min_score_floor,
            )
            if results:
                relaxed_min_score_fallback_used = True
                effective_min_score = min_score_floor
        context_sheet = format_context_sheet(
            query_text=query_text,
            path_prefix=resolved_prefix,
            top_k=top_k,
            results=results,
            max_chars=max_chars,
        )
        symbol_results = self._collect_context_symbols(
            query_text=query_text,
            project_id=project_id,
            language=language,
            embedding_dim=embedding_dim,
        )
        functional_context = assemble_functional_context(
            query_text=query_text,
            retrieval_results=results,
            symbol_results=symbol_results,
            path_prefix=resolved_prefix,
        )
        functional_context["tool_hints"] = _build_tool_hints(
            query_text=query_text,
            functional_context=functional_context,
            results=results,
            symbol_results=symbol_results,
        )
        meta: dict[str, Any] = {
            "project_id": project_id,
            "top_k": top_k,
            "path_prefix": resolved_prefix,
            "max_chars": max_chars,
            "min_score": effective_min_score,
            "auto_scope_fallback_used": auto_scope_fallback_used,
            "relaxed_min_score_fallback_used": relaxed_min_score_fallback_used,
            "initial_path_prefix": initial_resolved_prefix,
        }
        if not results:
            meta["no_match_reason"] = {
                "threshold_was": effective_min_score,
                "all_fallbacks_attempted": auto_scope_fallback_used or relaxed_min_score_fallback_used,
                "suggestion": (
                    "Try a shorter or more specific query. "
                    + ("Remove or broaden path_prefix. " if initial_resolved_prefix else "")
                    + "Check that the project has been indexed (list_projects / get_project_info)."
                ),
            }
        payload = {
            "functional_context": functional_context,
            "context": context,
            "context_sheet": context_sheet,
            "results": results,
            "symbol_results": symbol_results,
            "meta": meta,
        }
        self._log_activity(
            "query_out",
            {
                "tool": tool_name,
                "project_id": project_id,
                "path_prefix": resolved_prefix,
                "result_count": len(results),
                "has_results": bool(results),
            },
        )
        # Cache successful results only
        if results and cache_key:
            self._set_cache(cache_key, payload)
        return payload

    def _run_rag_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute rag_search tool returning raw retrieval data plus investigation hints."""
        base_payload = self._run_rag_context(args, tool_name="rag_search")
        results = base_payload.get("results") or []
        meta = dict(base_payload.get("meta") or {})
        functional_context = base_payload.get("functional_context") or {}
        tool_hints = functional_context.get("tool_hints") or {}
        symbol_follow_up = tool_hints.get("symbol_follow_up") or {}
        result_groups = _build_rag_search_result_groups(results)
        summary = {
            "result_count": len(results),
            "unique_file_count": len(result_groups),
            "top_score": max((float(item.get("score") or 0.0) for item in results), default=0.0),
            "auto_scope_fallback_used": bool(meta.get("auto_scope_fallback_used")),
            "coverage": _classify_rag_search_coverage(result_groups),
        }
        investigation_hints = _build_rag_search_hints(
            result_groups=result_groups,
            symbol_follow_up=symbol_follow_up,
            meta=meta,
        )
        return {
            "query": {
                "text": functional_context.get("query", {}).get("text"),
                "path_prefix": meta.get("path_prefix"),
                "top_k": meta.get("top_k"),
                "doc_type": args.get("doc_type") or self._config.default_doc_type,
                "language": args.get("language"),
            },
            "summary": summary,
            "result_groups": result_groups,
            "investigation_hints": investigation_hints,
            "results": results,
            "meta": {
                **meta,
                "search_mode": "investigation",
            },
        }

    def _run_rag_full_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute rag_full_context: rag_context + symbols + decision_points in one call."""
        base = self._run_rag_context(args, tool_name="rag_full_context")
        functional_context = base.get("functional_context") or {}
        meta = base.get("meta") or {}
        project_id = str(meta.get("project_id") or "")
        embedding_dim = self._default_embedding_dim

        include_symbols = parse_bool(str(args.get("include_symbols", True)))
        explain = parse_bool(str(args.get("explain", False)))

        # Symbols are already collected by rag_context; use them
        symbol_results = base.get("symbol_results") or []
        entry_points = functional_context.get("entry_points") or []
        core_files = functional_context.get("core_files") or []

        # Build decision_points from entry_points enriched with caller sites
        decision_points: list[dict[str, Any]] = []
        if project_id and entry_points:
            for ep in entry_points[:6]:
                name = str(ep.get("name") or "")
                if not name:
                    continue
                callers: list[dict[str, Any]] = []
                try:
                    pool = get_pool(self._default_dsn)
                    if pool is not None:
                        with pool.connection() as conn:
                            store = RagStore(conn, embedding_dim)
                            if store.has_symbol_refs_table():
                                callers = store.query_caller_sites(name, project_id, limit=10)
                    else:
                        conn = get_connection(self._default_dsn)
                        try:
                            store = RagStore(conn, embedding_dim)
                            if store.has_symbol_refs_table():
                                callers = store.query_caller_sites(name, project_id, limit=10)
                        finally:
                            conn.close()
                except Exception:
                    pass
                dp: dict[str, Any] = {
                    "symbol": name,
                    "kind": ep.get("kind"),
                    "path": ep.get("source_path"),
                    "lines": [ep.get("line_start"), ep.get("line_end")],
                    "signature": ep.get("signature"),
                }
                if callers:
                    dp["callers"] = callers
                    dp["impact_summary"] = f"Referenced in {len(callers)} location(s)"
                if explain:
                    matching_file = next(
                        (cf for cf in core_files if cf.get("source_path") == ep.get("source_path")),
                        None,
                    )
                    if matching_file:
                        dp["selection_reason"] = matching_file.get("selection_reason", "")
                        dp["functional_role"] = matching_file.get("functional_role", "")
                decision_points.append(dp)

        # Compute confidence as mean of max_score across core_files
        scores = [float(cf.get("max_score", 0.0)) for cf in core_files if cf.get("max_score")]
        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        # Check if a snapshot exists for this project
        snapshot_path = self._project_root / ".local" / "snapshots" / project_id / "snapshot_index.json"

        payload: dict[str, Any] = {
            "context": base.get("context"),
            "functional_context": functional_context,
            "symbols": symbol_results if include_symbols else [],
            "decision_points": decision_points,
            "confidence": confidence,
            "fallback_used": bool(
                meta.get("auto_scope_fallback_used") or meta.get("relaxed_min_score_fallback_used")
            ),
            "snapshot_available": snapshot_path.exists(),
            "meta": meta,
        }
        if not base.get("results"):
            payload["no_match_reason"] = meta.get("no_match_reason")
        return payload

    def _run_context_info(self) -> dict[str, Any]:
        """Describe scope and boundaries of llm-context MCP."""
        self._refresh_project_registry_if_changed()
        database_runtime = self._augment_database_runtime_summary(
            self._get_database_runtime_summary()
        )
        runtime_readiness = self._build_runtime_readiness(
            ready=self._ready.is_set(),
            database_runtime=database_runtime,
        )
        return {
            "server": "llm-context-mcp",
            "runtime_name": self._runtime_name,
            "config_path": self._config_path,
            "storage_target": self._build_storage_target_summary(),
            "database_runtime": database_runtime,
            "runtime_readiness": runtime_readiness,
            "purpose": "Retrieve context from indexed code and documents (RAG).",
            "write_enabled": self._config.write_enabled,
            "ingest_enabled": self._config.ingest_enabled,
            "project_count": self._project_registry.project_count(),
            "tool_map": {
                "working_context": ["rag_context", "map_work_item_to_codebase"],
                "inspection": ["rag_search", "symbol_search"],
                "discovery": ["context_info", "list_projects", "get_project_info"],
            },
            "quick_start": [
                {
                    "step": "1",
                    "tool": "context_info",
                    "reason": "understand roles, limits and available tools before choosing a workflow",
                },
                {
                    "step": "2",
                    "tool": "rag_context",
                    "reason": "get the main context package to start working on the code",
                },
                {
                    "step": "3",
                    "tool": "symbol_search",
                    "reason": "use it after rag_context when you need exact lines or signatures",
                },
            ],
            "search_levels": [
                {
                    "level": "wide",
                    "tool": "rag_search",
                    "goal": "broad exploration when you do not know the files or symbols yet",
                    "query_shape": "2-4 domain terms, avoiding overly generic or mixed words",
                    "example": "invoice export pipeline",
                    "next_step": "pick a file or a symbol from result_groups",
                },
                {
                    "level": "symbolic",
                    "tool": "symbol_search",
                    "goal": "precision on classes, methods, interfaces, endpoints or models",
                    "query_shape": "symbol name or technical prefix",
                    "example": "InvoiceExporter",
                    "next_step": "use source_path and line_start to open the exact code",
                },
                {
                    "level": "specific",
                    "tool": "rag_context",
                    "goal": "working context on a file or scope once the area is known",
                    "query_shape": "technical question with an explicit path_prefix or file",
                    "example": "query_text='invoice export status flow' path_prefix='src/billing/invoice_exporter.py'",
                    "next_step": "work on core_files and verify with symbol_search",
                },
            ],
            "tool_roles": {
                "rag_context": {
                    "role": "default working tool",
                    "use_when": [
                        "you need the most useful context package to work on the code quickly",
                        "you want entry points, core files and assembled context instead of raw hits only",
                    ],
                    "returns": [
                        "functional_context",
                        "entry_points",
                        "core_files",
                        "supporting_matches",
                        "assembled_context",
                        "tool_hints",
                    ],
                },
                "rag_search": {
                    "role": "targeted research and deep inspection",
                    "use_when": [
                        "you want to see the raw retrieval matches",
                        "you need to dig deeper, confirm or debug query coverage",
                    ],
                    "returns": ["results", "summary", "result_groups", "investigation_hints", "meta"],
                },
                "symbol_search": {
                    "role": "precision lookup",
                    "use_when": [
                        "you need the exact signature and lines of a symbol",
                        "you need to disambiguate technical names surfaced by rag_context",
                    ],
                    "returns": ["results", "count"],
                },
                "map_work_item_to_codebase": {
                    "role": "functional-to-codebase mapping",
                    "use_when": [
                        "you start from a ticket, work item or functional request",
                        "you want a structured repo_target, area, paths and implementation hint",
                    ],
                    "returns": [
                        "product_target",
                        "repo_target",
                        "area",
                        "feasibility",
                        "paths",
                        "implementation_hint",
                    ],
                },
            },
            "decision_guide": [
                {
                    "if": "you have a mixed multi-term query such as 'REST endpoint invoice export status'",
                    "use": ["rag_search wide", "symbol_search symbolic", "rag_context specific"],
                    "because": "first widen on domain terms, then pin down the symbol or file, finally ask for context on the right scope",
                },
                {
                    "if": "you start from a technical query and want the right context to work",
                    "use": ["rag_context"],
                    "because": "it is the main tool and returns the functional package, entry points and core files",
                },
                {
                    "if": "you start from a ticket, work item or functional request",
                    "use": ["map_work_item_to_codebase", "rag_context"],
                    "because": "first map the request onto the codebase, then open the working context",
                },
                {
                    "if": "you want to understand why retrieval picks certain hits or see the raw matches",
                    "use": ["rag_search"],
                    "because": "it returns raw hits, per-file groups and investigation hints without assembly",
                },
                {
                    "if": "you need a precise signature, exact lines or to disambiguate a symbol",
                    "use": ["symbol_search"],
                    "because": "it is the precision tool and follows the symbol hints surfaced by rag_context",
                },
                {
                    "if": "you do not know which project_id to use",
                    "use": ["list_projects", "get_project_info"],
                    "because": "you must choose the correct read-plane project scope first",
                },
            ],
            "capabilities": [
                "rag_context: formatted context for code and document analysis",
                "rag_search: raw semantic and keyword retrieval results",
                "map_work_item_to_codebase: structured mapping between a functional request and the codebase",
                "symbol_search: exact or prefix symbol lookup by name (class/function/method/...)",
                "list_projects: list of registered projects",
                "get_project_info: details of a registered project",
            ],
            "usage_notes": {
                "rag_context": (
                    "Main working tool: returns an assembled functional package by default."
                ),
                "rag_search": (
                    "Deep-dive/raw tool: returns raw hits, per-file groups and investigation hints."
                ),
                "map_work_item_to_codebase": (
                    "Structured tool to map functional requests or tickets to product, repository and area."
                ),
                "symbol_search": (
                    "Precision tool for symbol lookup; useful to disambiguate technical names and entry points."
                ),
            },
            "recommended_workflows": [
                {
                    "goal": "search in three levels",
                    "steps": [
                        "wide: rag_search with domain terms",
                        "symbolic: symbol_search on the surfaced names",
                        "specific: rag_context with path_prefix/file",
                    ],
                },
                {
                    "goal": "start working on a technical query",
                    "steps": ["context_info", "rag_context", "symbol_search (when exact lines or signatures are needed)"],
                },
                {
                    "goal": "dig into or debug retrieval",
                    "steps": ["rag_context", "rag_search", "symbol_search"],
                },
                {
                    "goal": "start from a ticket or functional request",
                    "steps": ["map_work_item_to_codebase", "rag_context", "symbol_search"],
                },
            ],
            "anti_patterns": [
                "do not use rag_search as the first tool when your goal is to start working on the code",
                "do not use symbol_search for broad exploration: start from rag_context or map_work_item_to_codebase",
                "do not expect context_info to perform retrieval: it helps choose the right tool, it does not fetch context",
            ],
            "boundaries": [
                "It is NOT a persistent operational memory system",
                "It does NOT replace llm-memory for operational decisions or preferences",
                "Read-plane queries always require an explicit project_id",
                "Ingest is not exposed as a standard MCP tool",
            ],
        }

    def _run_symbol_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute symbol_search tool."""
        name = args.get("name")
        if not name:
            raise ValueError("'name' is required for symbol_search")
        project_id = self._resolve_project_id(args, tool_name="symbol_search")
        kind = args.get("kind") or None
        language = args.get("language") or None
        exact = parse_bool(str(args.get("exact", False)))
        limit = _bounded_int_arg(args, "limit", default=20, maximum=self._max_symbol_limit)
        self._reject_client_embedder_overrides(args)
        embedding_dim = self._default_embedding_dim
        self._log_activity(
            "query_in",
            {
                "tool": "symbol_search",
                "project_id": project_id,
                "name": self._preview_value(name),
                "kind": kind,
                "language": language,
                "exact": exact,
                "limit": limit,
            },
        )

        results = self._query_symbols(
            name=name,
            project_id=project_id,
            kind=kind,
            language=language,
            exact=exact,
            limit=limit,
            embedding_dim=embedding_dim,
        )
        payload = {
            "query": {"name": name, "kind": kind, "language": language, "exact": exact},
            "count": len(results),
            "results": results,
        }
        self._log_activity(
            "query_out",
            {
                "tool": "symbol_search",
                "project_id": project_id,
                "name": self._preview_value(name),
                "result_count": len(results),
                "has_results": bool(results),
            },
        )
        return payload

    def _run_map_work_item_to_codebase(self, args: dict[str, Any]) -> dict[str, Any]:
        summary = str(args.get("summary") or "").strip()
        description = str(args.get("description") or "").strip()
        if not summary and not description:
            raise ValueError("summary or description is required")

        project_id = self._resolve_mapping_project_id(args)
        project_record = self._project_registry.get_project(project_id)
        mapping_query = "\n\n".join(part for part in [summary, description] if part).strip()
        rag_args = {
            "query_text": mapping_query,
            "project_id": project_id,
            "top_k": self._top_k_arg(args),
            "auto_scope": False,
            "format": "json",
        }
        if args.get("language"):
            rag_args["language"] = args.get("language")
        if args.get("doc_type"):
            rag_args["doc_type"] = args.get("doc_type")
        if args.get("path_prefix"):
            rag_args["path_prefix"] = args.get("path_prefix")

        rag_payload = self._run_rag_context(rag_args, tool_name="map_work_item_to_codebase")
        mapping = map_work_item_to_codebase(
            summary=summary,
            description=description,
            product_target_hint=args.get("product_target_hint"),
            project_record=project_record or type("ProjectStub", (), {"project_id": project_id, "root_path": project_id})(),
            functional_context=rag_payload.get("functional_context") or {},
            product_targets=self._project_setting(project_record, "product_targets"),
            product_aliases=self._project_setting(project_record, "product_aliases"),
        )
        mapping.update(
            {
                "ticket_key": str(args.get("ticket_key") or "").strip() or None,
                "project_id": project_id,
                "workspace_root": str(
                    args.get("workspace_root")
                    or getattr(project_record, "root_path", "")
                    or ""
                ).strip()
                or None,
            }
        )
        return mapping

    def _run_list_projects(self) -> dict[str, Any]:
        self._refresh_project_registry_if_changed()
        projects = [
            self._build_project_public_payload(project)
            for project in self._project_registry.list_projects()
        ]
        return {
            "count": len(projects),
            "write_enabled": self._config.write_enabled,
            "ingest_enabled": self._config.ingest_enabled,
            "projects": projects,
        }

    def _run_get_project_info(self, args: dict[str, Any]) -> dict[str, Any]:
        project_id = str(args.get("project_id") or "").strip()
        if not project_id:
            raise ValueError("'project_id' is required for get_project_info")
        self._refresh_project_registry_if_changed()
        project = self._project_registry.require_project(project_id)
        return self._build_project_public_payload(project)

    def _run_get_project_snapshot(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the pre-computed snapshot (CONTEXT.md + index) for a project."""
        project_id = str(args.get("project_id") or "").strip()
        if not project_id:
            raise ValueError("'project_id' is required for get_project_snapshot")
        self._validate_known_project(project_id)

        snapshot_dir = self._project_root / ".local" / "snapshots" / project_id
        index_path = snapshot_dir / "snapshot_index.json"
        context_md_path = snapshot_dir / "CONTEXT.md"

        if not snapshot_dir.exists() or not index_path.exists():
            return {
                "available": False,
                "project_id": project_id,
                "snapshot_dir": str(snapshot_dir),
                "message": (
                    f"No snapshot found for project '{project_id}'. "
                    "Generate one with: "
                    f"python cli.py export-snapshot --project-id {project_id} "
                    f"--dsn $LLM_CONTEXT_DSN --output .local/snapshots"
                ),
            }

        import json as _json
        try:
            snapshot_index = _json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            snapshot_index = {"error": str(exc)}

        context_md: Optional[str] = None
        part = str(args.get("part") or "all").strip().lower()
        if part in {"all", "context_md"} and context_md_path.exists():
            context_md = context_md_path.read_text(encoding="utf-8")

        symbols_catalog: Optional[list] = None
        functional_areas: Optional[list] = None
        top_files: Optional[list] = None

        if part == "all" or part == "symbols":
            sym_path = snapshot_dir / "symbols_catalog.json"
            if sym_path.exists():
                try:
                    symbols_catalog = _json.loads(sym_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        if part == "all" or part == "areas":
            areas_path = snapshot_dir / "functional_areas.json"
            if areas_path.exists():
                try:
                    functional_areas = _json.loads(areas_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        if part == "all" or part == "top_files":
            top_path = snapshot_dir / "top_files.json"
            if top_path.exists():
                try:
                    top_files = _json.loads(top_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        payload: dict[str, Any] = {
            "available": True,
            "project_id": project_id,
            "snapshot_dir": str(snapshot_dir),
            "snapshot_index": snapshot_index,
        }
        if context_md is not None:
            payload["context_md"] = context_md
        if symbols_catalog is not None:
            payload["symbols_catalog"] = symbols_catalog
        if functional_areas is not None:
            payload["functional_areas"] = functional_areas
        if top_files is not None:
            payload["top_files"] = top_files
        return payload

    def get_operational_status(self, *, ready: bool) -> dict[str, Any]:
        self._refresh_project_registry_if_changed()
        projects = [
            self._build_project_status_summary(project)
            for project in self._project_registry.list_projects()
        ]
        database_runtime = self._augment_database_runtime_summary(
            self._get_database_runtime_summary()
        )
        runtime_readiness = self._build_runtime_readiness(
            ready=ready,
            project_summaries=projects,
            database_runtime=database_runtime,
        )
        return {
            "status": "ready" if ready else "loading",
            "runtime_name": self._runtime_name,
            "config_path": self._config_path,
            "storage_target": self._build_storage_target_summary(),
            "database_runtime": database_runtime,
            "runtime_readiness": runtime_readiness,
            "project_manifest_dir": str(self._project_registry.manifest_dir),
            "ingest_enabled": self._config.ingest_enabled,
            "write_enabled": self._config.write_enabled,
            "project_count": len(projects),
            "projects": projects,
        }

    def _build_project_status_summary(self, project) -> dict[str, Any]:
        integrity = self._build_project_integrity(project)
        manifest = project.index_manifest.to_public_dict() if project.index_manifest is not None else None
        return {
            "project_id": project.project_id,
            "display_name": project.display_name,
            "ingest_enabled": project.ingest_enabled,
            "write_enabled": project.write_enabled,
            "last_ingest_status": project.last_ingest_status,
            "last_successful_ingest_at": project.last_successful_ingest_at,
            "index_version": project.index_version,
            "index_fingerprint": project.index_fingerprint,
            "index_manifest": manifest,
            "integrity": integrity,
        }

    def _build_project_public_payload(self, project) -> dict[str, Any]:
        payload = project.to_public_dict()
        payload["integrity"] = self._build_project_integrity(project)
        return payload

    def _build_runtime_readiness(
        self,
        *,
        ready: bool,
        project_summaries: Optional[list[dict[str, Any]]] = None,
        database_runtime: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if project_summaries is None:
            project_summaries = [
                self._build_project_status_summary(project)
                for project in self._project_registry.list_projects()
            ]
        if database_runtime is None:
            database_runtime = self._get_database_runtime_summary()
        database_runtime = self._augment_database_runtime_summary(database_runtime)

        warnings: list[str] = []
        blockers: list[str] = []
        recommended_actions: list[str] = []
        storage_target = self._build_storage_target_summary()

        def add_warning(value: str) -> None:
            if value and value not in warnings:
                warnings.append(value)

        def add_blocker(value: str) -> None:
            if value and value not in blockers:
                blockers.append(value)

        def add_action(value: str) -> None:
            if value and value not in recommended_actions:
                recommended_actions.append(value)

        checks: list[dict[str, Any]] = [
            {
                "name": "embedder_warmup",
                "status": "ok" if ready else "pending",
                "detail": (
                    "Embedder warmup completed."
                    if ready
                    else "Embedder warmup still in progress."
                ),
            }
        ]
        if not ready:
            add_blocker("embedder_warmup_in_progress")
            add_action("Wait for the embedder warmup to finish before using rag_context.")

        dedicated_candidate = bool(storage_target.get("dedicated_candidate"))
        deployment_hint = str(storage_target.get("deployment_hint") or "").strip() or "unknown"
        checks.append(
            {
                "name": "storage_target",
                "status": "ok" if dedicated_candidate else "warning",
                "detail": (
                    "Storage target is dedicated to this runtime."
                    if dedicated_candidate
                    else "Storage target is not clearly dedicated to this runtime."
                ),
            }
        )
        if not dedicated_candidate:
            add_warning("storage_target_is_not_clearly_dedicated")
            add_action("Use a database dedicated to llm-context to avoid ambiguity with other runtimes.")

        database_reachable = bool(database_runtime.get("reachable"))
        database_error = str(database_runtime.get("error") or "").strip()
        checks.append(
            {
                "name": "database_connection",
                "status": "ok" if database_reachable else "error",
                "detail": (
                    f"PostgreSQL database reachable: {database_runtime.get('database') or storage_target.get('database') or '(unknown)'}."
                    if database_reachable
                    else (
                        f"Database connection failed: {database_error}"
                        if database_error
                        else "Database connection failed."
                    )
                ),
            }
        )
        if not database_reachable:
            add_blocker("database_unreachable")
            add_action(_database_unreachable_action(deployment_hint))

        pgvector_available = database_runtime.get("pgvector_available")
        checks.append(
            {
                "name": "pgvector_extension",
                "status": (
                    "ok"
                    if pgvector_available is True
                    else ("error" if database_reachable else "pending")
                ),
                "detail": (
                    "pgvector extension available."
                    if pgvector_available is True
                    else (
                        "pgvector extension not found in the target database."
                        if database_reachable
                        else "pgvector check deferred until the database is reachable."
                    )
                ),
            }
        )
        if database_reachable and pgvector_available is not True:
            add_blocker("pgvector_extension_missing")
            add_action(_pgvector_missing_action(deployment_hint))

        schema_ready = database_runtime.get("schema_ready")
        required_tables = database_runtime.get("required_tables") or {}
        missing_tables = [
            name for name, present in required_tables.items() if not bool(present)
        ]
        checks.append(
            {
                "name": "database_schema",
                "status": (
                    "ok"
                    if schema_ready is True
                    else ("error" if database_reachable else "pending")
                ),
                "detail": (
                    "Schema v2 present on the target database."
                    if schema_ready is True
                    else (
                        "Schema v2 incomplete; missing tables: "
                        + ", ".join(missing_tables[:5])
                        if database_reachable and missing_tables
                        else (
                            "Schema v2 cannot be verified until the database is reachable."
                        )
                    )
                ),
            }
        )
        if database_reachable and schema_ready is not True:
            add_blocker("database_schema_not_initialized")
            add_action(_schema_missing_action(deployment_hint))

        project_count = len(project_summaries)
        registry_ready = project_count > 0
        registry_status = "ok" if registry_ready else "error"
        registry_detail = (
            f"{project_count} registered project(s)."
            if registry_ready
            else "No project registered in the registry."
        )
        checks.append(
            {
                "name": "project_registry",
                "status": registry_status,
                "detail": registry_detail,
            }
        )
        if not registry_ready:
            add_blocker("no_registered_projects")
            add_action("Register at least one project in the registry before using the read-plane or ingest.")

        integrity_counts: dict[str, int] = {}
        queryable_projects: list[str] = []
        indexing_projects: list[str] = []
        for item in project_summaries:
            integrity = item.get("integrity") or {}
            integrity_status = str(integrity.get("status") or "unknown").strip() or "unknown"
            integrity_counts[integrity_status] = integrity_counts.get(integrity_status, 0) + 1
            project_id = str(item.get("project_id") or "").strip()
            if integrity_status == "ok" and project_id:
                queryable_projects.append(project_id)
            elif integrity_status == "indexing" and project_id:
                indexing_projects.append(project_id)

        if queryable_projects:
            indexed_status = "ok"
            indexed_detail = (
                "Queryable projects: " + ", ".join(queryable_projects[:5])
            )
        elif indexing_projects:
            indexed_status = "warning"
            indexed_detail = (
                "Ingest in progress without a coherent project yet: "
                + ", ".join(indexing_projects[:5])
            )
        else:
            indexed_status = "error"
            indexed_detail = "No project is currently queryable."
        checks.append(
            {
                "name": "indexed_projects",
                "status": indexed_status,
                "detail": indexed_detail,
                "integrity_counts": integrity_counts,
            }
        )

        if integrity_counts.get("indexing"):
            add_warning("ingest_in_progress_without_stable_project")
            add_action("Wait for ingest to finish before considering the runtime stable.")
        if integrity_counts.get("not_indexed"):
            add_warning("registered_projects_are_not_indexed")
            add_action("Run ingest on the registered projects before using the runtime in production.")
        if integrity_counts.get("stale"):
            add_warning("registered_projects_have_stale_index_state")
            add_action("Align runtime state and manifest, or re-run ingest on stale projects.")
        if integrity_counts.get("unreliable"):
            add_warning("registered_projects_have_unreliable_index_state")
            add_action("Fix ingest or manifest errors before trusting the results.")

        if project_count > 0 and not queryable_projects and not indexing_projects:
            add_blocker("no_project_with_integrity_ok")

        ready_for_queries = ready and bool(queryable_projects) and not blockers
        if blockers:
            status = "blocked"
        elif warnings:
            status = "degraded" if ready else "blocked"
        elif ready_for_queries:
            status = "ready"
        else:
            status = "blocked"

        if ready_for_queries and status == "ready":
            summary = (
                f"Runtime ready: {len(queryable_projects)} coherent queryable project(s)."
            )
        elif ready_for_queries:
            summary = (
                f"Runtime queryable but with signals to check: {len(queryable_projects)} coherent project(s)."
            )
        elif indexing_projects:
            summary = "Runtime started but still waiting for a coherent project after ingest."
        elif project_count == 0:
            summary = "Runtime started but no verifiable project is ready for the read-plane."
        else:
            summary = "Runtime not ready: no project with integrity=ok."

        return {
            "status": status,
            "ready_for_queries": ready_for_queries,
            "summary": summary,
            "checks": checks,
            "project_integrity_counts": integrity_counts,
            "queryable_projects": queryable_projects,
            "blocking_reasons": blockers,
            "warnings": warnings,
            "recommended_actions": recommended_actions,
        }

    def _build_project_integrity(self, project) -> dict[str, Any]:
        manifest = project.index_manifest
        runtime_state = project.runtime_state
        runtime_target = self._build_storage_target_summary()
        reasons: list[str] = []
        status = "ok"

        runtime_status = str(runtime_state.last_ingest_status or "").strip().lower()
        manifest_status = (
            str(manifest.last_ingest_status or "").strip().lower()
            if manifest is not None
            else ""
        )

        runtime_has_index = bool(
            runtime_state.index_version
            or runtime_state.index_fingerprint
            or runtime_state.last_successful_ingest_at
            or runtime_status == "success"
        )
        manifest_has_index = bool(
            manifest is not None
            and (
                manifest.index_version
                or manifest.index_fingerprint
                or manifest.last_ingest_completed_at
                or manifest_status == "success"
            )
        )

        if runtime_status == "running" or manifest_status == "running":
            status = "indexing"

        if runtime_status == "failed":
            status = "unreliable"
            reasons.append("runtime_state_reports_failed_ingest")
        if manifest_status == "failed":
            status = "unreliable"
            reasons.append("index_manifest_reports_failed_ingest")
        if manifest is not None and manifest.last_error:
            status = "unreliable"
            reasons.append("index_manifest_has_last_error")

        if manifest is None:
            if runtime_has_index:
                status = "unreliable"
                reasons.append("runtime_state_has_index_but_manifest_missing")
            elif status != "indexing":
                status = "not_indexed"
                reasons.append("no_index_manifest_present")
            return {
                "status": status,
                "reasons": reasons,
                "runtime_status": runtime_state.last_ingest_status,
                "manifest_status": None,
                "store_target_match": None,
            }

        if runtime_status and manifest_status and runtime_status != manifest_status:
            status = "stale"
            reasons.append("runtime_and_manifest_status_mismatch")
        if manifest_has_index and not runtime_has_index:
            status = "stale"
            reasons.append("index_manifest_has_index_but_runtime_state_is_missing")
        if runtime_has_index and not manifest_has_index and status == "ok":
            status = "stale"
            reasons.append("runtime_state_has_index_but_manifest_is_incomplete")
        if (
            runtime_state.index_version
            and manifest.index_version
            and runtime_state.index_version != manifest.index_version
        ):
            status = "stale"
            reasons.append("runtime_and_manifest_index_version_mismatch")
        if (
            runtime_state.index_fingerprint
            and manifest.index_fingerprint
            and runtime_state.index_fingerprint != manifest.index_fingerprint
        ):
            status = "stale"
            reasons.append("runtime_and_manifest_index_fingerprint_mismatch")
        if (
            runtime_state.last_successful_ingest_at
            and manifest.last_ingest_completed_at
            and runtime_state.last_successful_ingest_at != manifest.last_ingest_completed_at
        ):
            status = "stale"
            reasons.append("runtime_and_manifest_timestamp_mismatch")

        manifest_target = manifest.store_target or {}
        runtime_database = str(runtime_target.get("database") or "").strip().lower()
        manifest_database = str(manifest_target.get("database") or "").strip().lower()
        runtime_dsn_fingerprint = str(runtime_target.get("dsn_fingerprint") or "").strip()
        manifest_dsn_fingerprint = str(manifest_target.get("dsn_fingerprint") or "").strip()
        store_target_match: Optional[bool] = None
        if runtime_database and manifest_database:
            store_target_match = runtime_database == manifest_database
        elif runtime_dsn_fingerprint and manifest_dsn_fingerprint:
            store_target_match = runtime_dsn_fingerprint == manifest_dsn_fingerprint
        if store_target_match is False:
            status = "stale"
            reasons.append("runtime_storage_target_differs_from_manifest")

        if status == "ok" and not (runtime_has_index or manifest_has_index):
            status = "not_indexed"
            reasons.append("manifest_present_without_completed_index")

        if status == "ok" and not reasons:
            reasons.append("runtime_and_manifest_are_coherent")

        return {
            "status": status,
            "reasons": reasons,
            "runtime_status": runtime_state.last_ingest_status,
            "manifest_status": manifest.last_ingest_status,
            "store_target_match": store_target_match,
        }

    def _build_storage_target_summary(self) -> dict[str, Any]:
        parsed = urlsplit(self._default_dsn)
        host = parsed.hostname or None
        database = parsed.path.lstrip("/") if parsed.path else ""
        database = database or None
        fingerprint = hashlib.sha256(
            self._default_dsn.encode("utf-8", errors="ignore")
        ).hexdigest()[:12]
        dedicated_candidate = True
        if not database or database.lower() in {"postgres", "template0", "template1"}:
            dedicated_candidate = False
        network_scope, deployment_hint = _classify_database_target(host)
        return {
            "name": self._store_target_name,
            "scheme": parsed.scheme or None,
            "host": host,
            "port": parsed.port,
            "database": database,
            "dsn_fingerprint": fingerprint,
            "dedicated_candidate": dedicated_candidate,
            "network_scope": network_scope,
            "deployment_hint": deployment_hint,
        }

    def _get_database_runtime_summary(self, *, refresh: bool = False) -> dict[str, Any]:
        now = monotonic()
        if not refresh and self._db_probe_ttl_seconds > 0:
            with self._db_probe_lock:
                if (
                    self._db_probe_cache is not None
                    and (now - self._db_probe_cache_at) < self._db_probe_ttl_seconds
                ):
                    return dict(self._db_probe_cache)

        runtime = inspect_database_runtime(
            self._default_dsn,
            connect_timeout=self._db_connect_timeout_seconds,
        )
        target_summary = self._build_storage_target_summary()
        runtime["host"] = target_summary.get("host")
        runtime["port"] = target_summary.get("port")
        runtime["network_scope"] = target_summary.get("network_scope")
        runtime["deployment_hint"] = target_summary.get("deployment_hint")
        with self._db_probe_lock:
            self._db_probe_cache = dict(runtime)
            self._db_probe_cache_at = now
        return dict(runtime)

    def _augment_database_runtime_summary(self, runtime: Optional[dict[str, Any]]) -> dict[str, Any]:
        payload = dict(runtime or {})
        target_summary = self._build_storage_target_summary()
        payload.setdefault("host", target_summary.get("host"))
        payload.setdefault("port", target_summary.get("port"))
        payload.setdefault("network_scope", target_summary.get("network_scope"))
        payload.setdefault("deployment_hint", target_summary.get("deployment_hint"))
        return payload

    def _resolve_project_id(self, args: dict[str, Any], *, tool_name: str) -> str:
        explicit_project_id = str(args.get("project_id") or "").strip()
        if not explicit_project_id:
            raise ValueError(
                f"{tool_name} requires explicit project_id. This is a project-scoped read-plane MCP tool; "
                "pass project_id explicitly and choose it via list_projects/get_project_info when needed."
            )
        return self._validate_known_project(explicit_project_id)

    def _resolve_mapping_project_id(self, args: dict[str, Any]) -> str:
        explicit_project_id = str(args.get("project_id") or "").strip()
        if not explicit_project_id:
            raise ValueError(
                "map_work_item_to_codebase requires explicit project_id. Choose it via list_projects/get_project_info first."
            )
        return self._validate_known_project(explicit_project_id)

    def _validate_known_project(self, project_id: str) -> str:
        self._refresh_project_registry_if_changed()
        return self._project_registry.require_project(project_id).project_id

    def _project_setting(self, project_record: Any, name: str) -> dict[str, Any]:
        """Per-project mapping from the registry, falling back to the global config value."""
        value = getattr(project_record, name, None) if project_record is not None else None
        if value:
            return dict(value)
        return dict(getattr(self._config, name, None) or {})

    def _registry_signature(self) -> tuple[Any, ...]:
        def mtime(path: Path) -> Optional[int]:
            try:
                return path.stat().st_mtime_ns
            except OSError:
                return None

        manifest_dir = self._project_registry.manifest_dir
        manifests: tuple[tuple[str, Optional[int]], ...] = ()
        if manifest_dir.is_dir():
            manifests = tuple(
                sorted((item.name, mtime(item)) for item in manifest_dir.glob("*.index_manifest.json"))
            )
        return (
            mtime(Path(self._config.projects_registry_path)),
            mtime(Path(self._config.projects_state_path)),
            manifests,
        )

    def _refresh_project_registry_if_changed(self) -> bool:
        """Reload the project registry when the registry, state or manifest files changed on disk."""
        with self._registry_lock:
            signature = self._registry_signature()
            if signature == self._registry_signature_value:
                return False
            try:
                registry = load_project_registry(
                    self._config.projects_registry_path,
                    self._config.projects_state_path,
                )
            except Exception as exc:
                sys.stderr.write(f"[WARN] Project registry reload failed, keeping previous registry: {exc}\n")
                return False
            self._project_registry = registry
            self._registry_signature_value = signature
            sys.stderr.write("[INFO] Project registry reloaded from disk\n")
            return True

    def _query_symbols(
        self,
        *,
        name: str,
        project_id: str,
        kind: Optional[str],
        language: Optional[str],
        exact: bool,
        limit: int,
        embedding_dim: int,
    ) -> list[dict[str, Any]]:
        pool = get_pool(self._default_dsn)
        if pool is not None:
            try:
                with pool.connection() as conn:
                    store = RagStore(conn, embedding_dim)
                    return store.query_symbols(
                        name=name,
                        repo_id=project_id,
                        kind=kind,
                        language=language,
                        exact=exact,
                        limit=limit,
                    )
            except Exception:
                pass

        conn = get_connection(self._default_dsn)
        try:
            store = RagStore(conn, embedding_dim)
            return store.query_symbols(
                name=name,
                repo_id=project_id,
                kind=kind,
                language=language,
                exact=exact,
                limit=limit,
            )
        finally:
            conn.close()

    def _collect_context_symbols(
        self,
        *,
        query_text: Any,
        project_id: str,
        language: Optional[str],
        embedding_dim: int,
    ) -> list[dict[str, Any]]:
        candidates = _derive_symbol_candidates(query_text)
        if not candidates:
            return []
        collected: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            try:
                matches = self._query_symbols(
                    name=candidate,
                    project_id=project_id,
                    kind=None,
                    language=language,
                    exact=False,
                    limit=5,
                    embedding_dim=embedding_dim,
                )
            except Exception:
                continue
            for item in matches:
                key = (
                    str(item.get("source_path") or ""),
                    str(item.get("name") or ""),
                    str(item.get("kind") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                collected.append(item)
        return collected

    def _validate_query_embedding(
        self,
        query_embedding: Any,
        embedding_dim: int,
    ) -> None:
        if query_embedding is None:
            return
        if not isinstance(query_embedding, list):
            raise ValueError("query_embedding must be an array of numeric values")

        item_count = len(query_embedding)
        if item_count > self._max_query_embedding_items:
            raise ValueError(
                "query_embedding exceeds maximum allowed length "
                f"({item_count} > {self._max_query_embedding_items})"
            )
        if item_count != embedding_dim:
            raise ValueError(
                f"query_embedding length mismatch: {item_count} != {embedding_dim}"
            )
        for item in query_embedding:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("query_embedding must contain only numeric values")

    def _top_k_arg(self, args: dict[str, Any]) -> int:
        return _bounded_int_arg(args, "top_k", default=8, maximum=self._max_top_k)

    def _max_chars_arg(self, args: dict[str, Any]) -> int:
        return _bounded_int_arg(
            args,
            "max_chars",
            default=min(self._default_max_chars, self._max_chars_limit),
            maximum=self._max_chars_limit,
        )

    def _reject_client_embedder_overrides(self, args: dict[str, Any]) -> None:
        """Embedder, model and dimension are server-side settings only.

        A client value equal to the server setting is ignored for backward
        compatibility; any other value is rejected so that a client can never
        make the server load an arbitrary model.
        """
        server_values = {
            "embedder": str(self._default_embedder).strip().lower(),
            "local_model": str(self._default_local_model).strip(),
            "embedding_dim": str(self._default_embedding_dim),
        }
        for key, server_value in server_values.items():
            if key not in args or args[key] is None:
                continue
            client_value = str(args[key]).strip()
            if key == "embedder":
                client_value = client_value.lower()
            if client_value != server_value:
                raise ValueError(
                    f"'{key}' is a server-side setting and cannot be overridden by the client"
                )
            sys.stderr.write(
                f"[WARN] Ignoring deprecated client argument '{key}' (matches server configuration)\n"
            )

    def _get_embedder(self) -> Embedder:
        """Get or create the server-configured embedder."""
        name = str(self._default_embedder).strip().lower()
        local_model = self._default_local_model
        embedding_dim = self._default_embedding_dim
        cache_key = f"{name}|{local_model}|{embedding_dim}"

        with self._embedder_lock:
            if cache_key in self._embedder_cache:
                return self._embedder_cache[cache_key]

            sys.stderr.write(f"[INFO] Loading embedder: {cache_key}...\n")
            if name == "local-hash":
                embedder = LocalHashEmbedder(embedding_dim)
            elif name in {"local-st", "sentence-transformers", "local"}:
                try:
                    embedder = LocalSentenceTransformerEmbedder(model_name=local_model)
                except Exception as exc:
                    if not self._allow_embedder_fallback:
                        raise
                    fallback_name = self._fallback_embedder
                    if fallback_name != "local-hash":
                        raise RuntimeError(
                            f"Embedder fallback '{fallback_name}' unsupported"
                        ) from exc
                    sys.stderr.write(
                        f"[WARN] local-st unavailable ({exc}); fallback to local-hash dim={embedding_dim}\n"
                    )
                    embedder = LocalHashEmbedder(embedding_dim)
            elif name == "dummy":
                embedder = DummyEmbedder()
            else:
                raise ValueError(f"Unknown embedder: {name}")

            self._embedder_cache[cache_key] = embedder
            sys.stderr.write(f"[INFO] Embedder loaded: {cache_key}\n")
            return embedder


def _bounded_int_arg(args: dict[str, Any], key: str, *, default: int, maximum: int) -> int:
    raw = args.get(key)
    if raw is None:
        return max(1, min(default, maximum))
    if isinstance(raw, bool):
        raise ValueError(f"'{key}' must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{key}' must be an integer") from exc
    return max(1, min(value, maximum))


def tool_rag_context() -> dict[str, Any]:
    """Return the rag_context tool schema."""
    return {
        "name": "rag_context",
        "description": (
            "Main read-plane MCP tool: returns a functional context package assembled from "
            "indexed code and documents, designed to help an agent start working on the code "
            "right away. Includes entry points, core files, supporting matches, assembled "
            "context and hints for follow-up tools. Use it for technical context retrieval only; "
            "it does not store operational memories and does not run ingest or index refresh. "
            "Always requires an explicit project_id and never uses an implicit default. For "
            "write_enabled/ingest_enabled, index refresh and ingest-plane operations use "
            "context_info and the operational CLI."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "query_embedding": {"type": "array", "items": {"type": "number"}},
                "project_id": {
                    "type": "string",
                    "description": (
                        "Retrieval project scope. Always required; must reference a project registered in the registry."
                    ),
                },
                "top_k": {"type": "integer"},
                "path_prefix": {"type": "string"},
                "file": {"type": "string"},
                "auto_scope": {"type": "boolean"},
                "max_chars": {"type": "integer"},
                "doc_type": {"type": "string"},
                "language": {"type": "string"},
                "format": {
                    "type": "string",
                    "description": "functional (default), legacy, sheet, or json",
                },
            },
        },
    }


def tool_rag_full_context() -> dict[str, Any]:
    """Return the rag_full_context tool schema."""
    base = tool_rag_context()
    base["name"] = "rag_full_context"
    base["description"] = (
        "All-in-one read-plane MCP tool: runs rag_context and automatically adds symbols, "
        "decision_points (where to act and estimated impact) and caller sites for the entry "
        "points found. Ideal for code change tasks where you need to know right away which "
        "lines to touch and what may break. Returns confidence, fallback_used and "
        "snapshot_available. Always requires an explicit project_id."
    )
    props = dict(base["inputSchema"]["properties"])
    props["include_symbols"] = {
        "type": "boolean",
        "description": "Include symbol lookup for core files (default: true)",
    }
    props["explain"] = {
        "type": "boolean",
        "description": "Include selection_reason and functional_role in decision_points (default: false)",
    }
    base["inputSchema"] = {"type": "object", "properties": props}
    return base


def tool_rag_search() -> dict[str, Any]:
    """Return the rag_search tool schema."""
    tool = tool_rag_context()
    tool["name"] = "rag_search"
    tool["description"] = (
        "Deep-dive read-plane MCP tool: returns raw RAG matches over indexed code and "
        "documents without context assembly. Use it for targeted research, retrieval "
        "debugging, spot checks, or when you need the raw results beyond the main "
        "rag_context package. It is not an operational memory store and does not run ingest "
        "or index refresh. Always requires an explicit project_id and never uses an implicit default."
    )
    tool["inputSchema"]["properties"]["format"]["description"] = "investigation (default) or json"
    return tool


def tool_map_work_item_to_codebase() -> dict[str, Any]:
    return {
        "name": "map_work_item_to_codebase",
        "description": (
            "Maps a functional request or ticket onto the indexed codebase and returns a "
            "structured payload with repo_target, area, feasibility, implementation hint, "
            "relevant paths and blockers. Uses the read-plane functional context engine."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["summary"],
            "properties": {
                "ticket_key": {"type": "string"},
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "product_target_hint": {"type": "string"},
                "project_id": {
                    "type": "string",
                    "description": "Explicit project scope. Always required.",
                },
                "path_prefix": {"type": "string"},
                "doc_type": {"type": "string"},
                "language": {"type": "string"},
                "top_k": {"type": "integer"},
            },
        },
    }


def tool_context_info() -> dict[str, Any]:
    """Return tool schema describing llm-context purpose and boundaries."""
    return {
        "name": "context_info",
        "description": (
            "Server discovery tool: describes purpose, limits, available tools, recommended "
            "roles and usage workflows of the llm-context MCP server."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    }


def tool_list_projects() -> dict[str, Any]:
    return {
        "name": "list_projects",
        "description": "Lists the registered projects and their ingest/index operational status.",
        "inputSchema": {"type": "object", "properties": {}},
    }


def tool_get_project_info() -> dict[str, Any]:
    return {
        "name": "get_project_info",
        "description": "Returns the operational details of a registered project.",
        "inputSchema": {
            "type": "object",
            "required": ["project_id"],
            "properties": {
                "project_id": {"type": "string"},
            },
        },
    }


def tool_symbol_search() -> dict[str, Any]:
    """Return the symbol_search tool schema."""
    return {
        "name": "symbol_search",
        "description": (
            "Searches symbols (class, function, method, interface, enum, struct, type) by name "
            "in the indexed code. It is the precision tool to use after rag_context when you "
            "need to disambiguate technical names, find exact signatures or lines, or confirm "
            "specific entry points. It is a read-plane tool: it does not update the index and "
            "does not perform ingest-plane operations. Always requires an explicit project_id. "
            "Returns line_start, line_end, signature and file path."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name (or prefix) to search for",
                },
                "kind": {
                    "type": "string",
                    "description": "Kind filter: class | function | method | interface | enum | struct | type",
                },
                "project_id": {
                    "type": "string",
                    "description": (
                        "Symbol search project scope. Always required; must reference a project registered in the registry."
                    ),
                },
                "language": {
                    "type": "string",
                    "description": "Language filter: python | javascript | typescript | csharp",
                },
                "exact": {
                    "type": "boolean",
                    "description": "If true: exact match (case-insensitive); if false (default): prefix match",
                },
                "limit": {"type": "integer"},
            },
        },
    }


def tool_get_project_snapshot() -> dict[str, Any]:
    return {
        "name": "get_project_snapshot",
        "description": (
            "Loads the pre-computed snapshot of a project: CONTEXT.md (orientation map of the "
            "whole codebase), symbol index and functional areas. "
            "Use it as the first call of a session to get oriented without spending live queries. "
            "The snapshot is generated offline with: "
            "python cli.py export-snapshot --project-id <id> --dsn $DSN --output .local/snapshots"
        ),
        "inputSchema": {
            "type": "object",
            "required": ["project_id"],
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Registered project id",
                },
                "part": {
                    "type": "string",
                    "description": (
                        "Part to load: 'all' (default), 'context_md', "
                        "'symbols', 'areas', 'top_files'"
                    ),
                },
            },
        },
    }


def format_tool_text(name: str, args: dict[str, Any], payload: dict[str, Any]) -> str:
    """Format tool output based on format hint."""
    format_hint = str(args.get("format", "") or "").strip().lower()
    if name == "get_project_snapshot":
        if not payload.get("available"):
            return str(payload.get("message", "Snapshot not available."))
        # Return context_md as primary text if present, else JSON
        context_md = payload.get("context_md")
        if context_md and format_hint not in {"json", "full"}:
            meta = payload.get("snapshot_index") or {}
            header = (
                f"[Snapshot: {payload.get('project_id')} — "
                f"generated {meta.get('generated_at', 'unknown')} — "
                f"{meta.get('indexed_documents', '?')} docs · "
                f"{meta.get('indexed_symbols', '?')} symbols]\n\n"
            )
            return header + context_md
        return json.dumps(payload, indent=2, ensure_ascii=True, cls=UUIDEncoder)
    if name == "rag_context":
        if format_hint in {"json", "full"}:
            return json.dumps(payload, indent=2, ensure_ascii=True, cls=UUIDEncoder)
        if format_hint in {"legacy", "text"}:
            return str(payload.get("context", ""))
        if format_hint in {"sheet", "context-sheet"}:
            return str(payload.get("context_sheet", ""))
        return format_functional_context_text(payload.get("functional_context") or {})
    if name == "rag_search":
        if format_hint in {"json", "full"}:
            return json.dumps(payload, indent=2, ensure_ascii=True, cls=UUIDEncoder)
        return format_rag_search_text(payload)
    if name == "context_info":
        if format_hint in {"json", "full"}:
            return json.dumps(payload, indent=2, ensure_ascii=True, cls=UUIDEncoder)
        return format_context_info_text(payload)
    return json.dumps(payload, indent=2, ensure_ascii=True, cls=UUIDEncoder)


def format_context_sheet(
    query_text: Optional[str],
    path_prefix: Optional[str],
    top_k: int,
    results: list[dict[str, Any]],
    max_chars: int,
) -> str:
    """Format results as a structured context sheet."""
    safe_query = (query_text or "").strip() or "(empty)"
    lines: list[str] = [
        "CONTEXT",
        f"query: {safe_query}",
        f"scope: {path_prefix or '(none)'}",
        f"top_k: {top_k}",
        "",
        "MATCHES",
    ]
    for index, item in enumerate(results, start=1):
        source_path = item.get("source_path", "")
        chunk_index = item.get("chunk_index", "")
        score = float(item.get("score", 0.0))
        line_start = item.get("line_start")
        line_end = item.get("line_end")
        snippet = (item.get("snippet") or item.get("text") or "").strip()
        if not snippet:
            continue
        if line_start and line_end:
            lines.append(
                f"{index}. {source_path} #L{line_start}-L{line_end} score={score:.4f}"
            )
        else:
            lines.append(
                f"{index}. {source_path} #chunk={chunk_index} score={score:.4f}"
            )
        lines.append(snippet)
        lines.append("")
    sheet = "\n".join(lines).strip()
    if max_chars > 0 and len(sheet) > max_chars:
        return sheet[: max_chars - 3] + "..."
    return sheet


def format_functional_context_text(payload: dict[str, Any]) -> str:
    query = payload.get("query") or {}
    summary = payload.get("summary") or {}
    entry_points = payload.get("entry_points") or []
    core_files = payload.get("core_files") or []
    supporting_matches = payload.get("supporting_matches") or []
    tool_hints = payload.get("tool_hints") or {}
    assembled_context = str(payload.get("assembled_context") or "").strip()

    lines: list[str] = [
        "FUNCTIONAL CONTEXT",
        f"query: {str(query.get('text') or '(empty)')}",
        f"scope: {str(query.get('path_prefix') or '(none)')}",
        (
            "summary: "
            f"core_files={int(summary.get('core_file_count') or 0)} "
            f"supporting_matches={int(summary.get('supporting_match_count') or 0)} "
            f"symbol_hits={int(summary.get('symbol_hit_count') or 0)}"
        ),
        "",
    ]
    if entry_points:
        lines.append("ENTRY POINTS")
        for item in entry_points:
            lines.append(
                f"- {item.get('kind') or 'symbol'} {item.get('name') or '(unknown)'} "
                f"{item.get('source_path') or ''} "
                f"#L{item.get('line_start')}-{item.get('line_end')}".rstrip()
            )
        lines.append("")
    if core_files:
        lines.append("CORE FILES")
        for item in core_files:
            lines.append(
                f"- {item.get('source_path')} score={float(item.get('aggregate_score', 0.0)):.4f} "
                f"matches={int(item.get('match_count') or 0)} symbols={len(item.get('symbol_hits') or [])}"
            )
        lines.append("")
    if supporting_matches:
        lines.append("SUPPORTING MATCHES")
        for item in supporting_matches[:5]:
            lines.append(
                f"- {item.get('source_path')} #L{item.get('line_start')}-{item.get('line_end')} "
                f"score={float(item.get('score', 0.0)):.4f}"
            )
        lines.append("")
    symbol_follow_up = tool_hints.get("symbol_follow_up") or {}
    suggested_queries = symbol_follow_up.get("suggested_queries") or []
    if suggested_queries:
        lines.append("SYMBOL FOLLOW-UP")
        for item in suggested_queries[:5]:
            name = item.get("name") or "(unknown)"
            kind = item.get("kind") or "symbol"
            role = item.get("source_role") or "supporting"
            source_path = item.get("source_path") or ""
            exact = bool(item.get("exact", False))
            reason = item.get("reason") or ""
            line = f"- {kind} {name} exact={str(exact).lower()} role={role}"
            if source_path:
                line += f" path={source_path}"
            if reason:
                line += f" reason={reason}"
            lines.append(line)
        lines.append("")
    follow_up_tools = tool_hints.get("recommended_follow_up") or []
    if follow_up_tools:
        lines.append("FOLLOW-UP TOOLS")
        for item in follow_up_tools:
            tool_name = item.get("tool") or "tool"
            reason = item.get("reason") or ""
            suggested_names = item.get("suggested_names") or []
            line = f"- {tool_name}: {reason}".rstrip()
            if suggested_names:
                line += f" suggested_names={','.join(str(name) for name in suggested_names)}"
            lines.append(line)
        lines.append("")
    if assembled_context:
        lines.append("ASSEMBLED CONTEXT")
        lines.append(assembled_context)
    return "\n".join(lines).strip()


def format_rag_search_text(payload: dict[str, Any]) -> str:
    query = payload.get("query") or {}
    summary = payload.get("summary") or {}
    result_groups = payload.get("result_groups") or []
    investigation_hints = payload.get("investigation_hints") or {}
    results = payload.get("results") or []

    lines: list[str] = [
        "RAG SEARCH",
        f"query: {str(query.get('text') or '(empty)')}",
        f"scope: {str(query.get('path_prefix') or '(none)')}",
        (
            "summary: "
            f"results={int(summary.get('result_count') or 0)} "
            f"files={int(summary.get('unique_file_count') or 0)} "
            f"coverage={summary.get('coverage') or 'unknown'} "
            f"top_score={float(summary.get('top_score') or 0.0):.4f}"
        ),
        "",
    ]
    if result_groups:
        lines.append("TOP FILES")
        for item in result_groups[:5]:
            line_spans = item.get("line_spans") or []
            line = (
                f"- {item.get('source_path')} hits={int(item.get('hit_count') or 0)} "
                f"top_score={float(item.get('top_score') or 0.0):.4f}"
            )
            if line_spans:
                line += f" lines={','.join(str(span) for span in line_spans)}"
            lines.append(line)
        lines.append("")
    suggested_files = investigation_hints.get("suggested_files") or []
    suggested_symbols = investigation_hints.get("suggested_symbol_queries") or []
    if suggested_files or suggested_symbols:
        lines.append("INVESTIGATION HINTS")
        for source_path in suggested_files[:4]:
            lines.append(f"- narrow_to_file: {source_path}")
        for item in suggested_symbols[:4]:
            name = item.get("name") or "(unknown)"
            kind = item.get("kind") or "symbol"
            exact = bool(item.get("exact", False))
            lines.append(f"- symbol_search: {kind} {name} exact={str(exact).lower()}")
        lines.append("")
    if results:
        lines.append("RAW MATCHES")
        for index, item in enumerate(results[:5], start=1):
            source_path = item.get("source_path") or "(unknown)"
            score = float(item.get("score") or 0.0)
            line_start = item.get("line_start")
            line_end = item.get("line_end")
            snippet = str(item.get("snippet") or item.get("text") or "").strip()
            if line_start and line_end:
                lines.append(f"{index}. {source_path} #L{line_start}-L{line_end} score={score:.4f}")
            else:
                lines.append(f"{index}. {source_path} score={score:.4f}")
            if snippet:
                lines.append(snippet)
            lines.append("")
    return "\n".join(lines).strip()


def format_context_info_text(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "CONTEXT INFO",
        f"server: {payload.get('server') or 'llm-context-mcp'}",
        f"runtime_name: {payload.get('runtime_name') or 'default'}",
        "",
    ]

    runtime_readiness = payload.get("runtime_readiness") or {}
    if runtime_readiness:
        lines.append("RUNTIME READINESS")
        lines.append(
            f"- status: {runtime_readiness.get('status') or 'unknown'} "
            f"ready_for_queries={str(bool(runtime_readiness.get('ready_for_queries'))).lower()}"
        )
        summary = runtime_readiness.get("summary") or ""
        if summary:
            lines.append(f"- summary: {summary}")
        for item in runtime_readiness.get("blocking_reasons") or []:
            lines.append(f"- blocker: {item}")
        for item in runtime_readiness.get("warnings") or []:
            lines.append(f"- warning: {item}")
        for item in (runtime_readiness.get("recommended_actions") or [])[:5]:
            lines.append(f"- action: {item}")
        lines.append("")

    database_runtime = payload.get("database_runtime") or {}
    if database_runtime:
        lines.append("DATABASE RUNTIME")
        lines.append(
            "- reachable: "
            + str(bool(database_runtime.get("reachable"))).lower()
            + f" database={database_runtime.get('database') or '(unknown)'}"
        )
        if database_runtime.get("deployment_hint"):
            lines.append(
                f"- deployment_hint: {database_runtime.get('deployment_hint')} "
                f"network_scope={database_runtime.get('network_scope') or 'unknown'}"
            )
        if database_runtime.get("server_version"):
            lines.append(f"- server_version: {database_runtime.get('server_version')}")
        if database_runtime.get("pgvector_available") is not None:
            lines.append(
                "- pgvector_available: "
                + str(bool(database_runtime.get("pgvector_available"))).lower()
            )
        if database_runtime.get("schema_ready") is not None:
            lines.append(
                "- schema_ready: "
                + str(bool(database_runtime.get("schema_ready"))).lower()
            )
        if database_runtime.get("error"):
            lines.append(f"- error: {database_runtime.get('error')}")
        lines.append("")

    quick_start = payload.get("quick_start") or []
    if quick_start:
        lines.append("QUICK START")
        for item in quick_start:
            step = item.get("step") or "step"
            tool = item.get("tool") or "tool"
            reason = item.get("reason") or ""
            lines.append(f"- {step}: {tool} {reason}".rstrip())
        lines.append("")

    search_levels = payload.get("search_levels") or []
    if search_levels:
        lines.append("SEARCH LEVELS")
        for item in search_levels:
            level = item.get("level") or "level"
            tool = item.get("tool") or "tool"
            goal = item.get("goal") or ""
            query_shape = item.get("query_shape") or ""
            example = item.get("example") or ""
            next_step = item.get("next_step") or ""
            lines.append(f"- {level}: {tool} - {goal}".rstrip())
            if query_shape:
                lines.append(f"  query: {query_shape}")
            if example:
                lines.append(f"  example: {example}")
            if next_step:
                lines.append(f"  next: {next_step}")
        lines.append("")

    decision_guide = payload.get("decision_guide") or []
    if decision_guide:
        lines.append("DECISION GUIDE")
        for item in decision_guide:
            condition = item.get("if") or ""
            use = ", ".join(str(tool) for tool in item.get("use") or [])
            because = item.get("because") or ""
            lines.append(f"- if {condition} -> use {use}; {because}".rstrip())
        lines.append("")

    anti_patterns = payload.get("anti_patterns") or []
    if anti_patterns:
        lines.append("ANTI-PATTERNS")
        for item in anti_patterns:
            lines.append(f"- {item}")
        lines.append("")

    recommended_workflows = payload.get("recommended_workflows") or []
    if recommended_workflows:
        lines.append("WORKFLOWS")
        for item in recommended_workflows:
            goal = item.get("goal") or "workflow"
            steps = " -> ".join(str(step) for step in item.get("steps") or [])
            lines.append(f"- {goal}: {steps}")
        lines.append("")

    return "\n".join(lines).strip()


def _build_tool_hints(
    *,
    query_text: Any,
    functional_context: dict[str, Any],
    results: list[dict[str, Any]],
    symbol_results: list[dict[str, Any]],
) -> dict[str, Any]:
    symbol_follow_up = _build_symbol_follow_up(
        query_text=query_text,
        functional_context=functional_context,
        symbol_results=symbol_results,
    )
    suggested_queries = symbol_follow_up.get("suggested_queries") or []
    suggested_names = symbol_follow_up.get("suggested_names") or []

    symbol_reason = (
        "Use symbol_search to confirm signatures, exact lines and relevant entry points."
        if suggested_queries or symbol_results
        else "Use symbol_search to disambiguate technical names or verify a specific symbol."
    )
    raw_reason = (
        "Use rag_search to see raw matches, explore query variants or investigate "
        "beyond the main package."
        if results
        else "Use rag_search with broader queries when rag_context does not return enough signal."
    )

    return {
        "primary_tool": "rag_context",
        "symbol_follow_up": symbol_follow_up,
        "recommended_follow_up": [
            {
                "tool": "symbol_search",
                "reason": symbol_reason,
                "suggested_names": suggested_names,
                "suggested_queries": suggested_queries,
                "focus_paths": symbol_follow_up.get("focus_paths") or [],
            },
            {
                "tool": "rag_search",
                "reason": raw_reason,
            },
            {
                "tool": "context_info",
                "reason": "Use context_info to review roles, limits and recommended workflows of the MCP tools.",
            },
        ],
    }


def _build_rag_search_result_groups(results: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for item in results:
        source_path = str(item.get("source_path") or "").strip()
        if not source_path:
            continue
        group = groups.setdefault(
            source_path,
            {
                "source_path": source_path,
                "hit_count": 0,
                "top_score": 0.0,
                "line_spans": [],
                "sample_snippet": "",
            },
        )
        group["hit_count"] += 1
        score = float(item.get("score") or 0.0)
        group["top_score"] = max(float(group.get("top_score") or 0.0), score)
        line_start = item.get("line_start")
        line_end = item.get("line_end")
        if line_start and line_end:
            span = f"L{line_start}-L{line_end}"
            if span not in group["line_spans"] and len(group["line_spans"]) < 3:
                group["line_spans"].append(span)
        if not group["sample_snippet"]:
            group["sample_snippet"] = str(item.get("snippet") or item.get("text") or "").strip()

    ranked = sorted(
        groups.values(),
        key=lambda item: (
            -int(item.get("hit_count") or 0),
            -float(item.get("top_score") or 0.0),
            str(item.get("source_path") or ""),
        ),
    )
    return ranked[:limit]


def _classify_rag_search_coverage(result_groups: list[dict[str, Any]]) -> str:
    if not result_groups:
        return "no_hits"
    if len(result_groups) == 1:
        return "single_hotspot"
    if len(result_groups) <= 3:
        return "focused_multi_file"
    return "broad_spread"


def _build_rag_search_hints(
    *,
    result_groups: list[dict[str, Any]],
    symbol_follow_up: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    suggested_files = [str(item.get("source_path") or "") for item in result_groups if item.get("source_path")]
    suggested_symbol_queries = symbol_follow_up.get("suggested_queries") or []
    recommended_next_steps: list[str] = []
    if result_groups:
        recommended_next_steps.append("Narrow down to one of the top files to confirm the right retrieval spot.")
    if meta.get("auto_scope_fallback_used"):
        recommended_next_steps.append("The initial auto-scope did not hold: try a more explicit query or a manual scope.")
    if suggested_symbol_queries:
        recommended_next_steps.append("Open the suggested symbols with symbol_search to verify signatures and exact lines.")
    if not result_groups:
        recommended_next_steps.append("Broaden the query or remove overly narrow filters to recover raw signal.")

    return {
        "mode": "raw_investigation",
        "suggested_files": suggested_files[:4],
        "suggested_symbol_queries": suggested_symbol_queries[:4],
        "recommended_next_steps": recommended_next_steps,
    }


def _build_symbol_follow_up(
    *,
    query_text: Any,
    functional_context: dict[str, Any],
    symbol_results: list[dict[str, Any]],
) -> dict[str, Any]:
    core_files = functional_context.get("core_files") or []
    suggested_queries: list[dict[str, Any]] = []
    suggested_names: list[str] = []
    focus_paths: list[str] = []
    seen_names: set[str] = set()

    for core_file in core_files:
        source_path = str(core_file.get("source_path") or "").strip()
        role = str(core_file.get("functional_role") or "supporting").strip() or "supporting"
        if source_path and source_path not in focus_paths:
            focus_paths.append(source_path)
        for symbol in core_file.get("symbol_hits") or []:
            name = str(symbol.get("name") or "").strip()
            if not name:
                continue
            lower = name.lower()
            if lower in seen_names:
                continue
            seen_names.add(lower)
            suggested_names.append(name)
            suggested_queries.append(
                {
                    "name": name,
                    "kind": str(symbol.get("kind") or "").strip() or None,
                    "exact": True,
                    "source_path": source_path or str(symbol.get("source_path") or "").strip() or None,
                    "source_role": role,
                    "reason": _symbol_follow_up_reason(role=role, kind=symbol.get("kind")),
                }
            )
            if len(suggested_queries) >= 6:
                break
        if len(suggested_queries) >= 6:
            break

    if not suggested_queries:
        for symbol in symbol_results:
            name = str(symbol.get("name") or "").strip()
            if not name:
                continue
            lower = name.lower()
            if lower in seen_names:
                continue
            seen_names.add(lower)
            source_path = str(symbol.get("source_path") or "").strip()
            if source_path and source_path not in focus_paths:
                focus_paths.append(source_path)
            suggested_names.append(name)
            suggested_queries.append(
                {
                    "name": name,
                    "kind": str(symbol.get("kind") or "").strip() or None,
                    "exact": True,
                    "source_path": source_path or None,
                    "source_role": "supporting",
                    "reason": _symbol_follow_up_reason(role="supporting", kind=symbol.get("kind")),
                }
            )
            if len(suggested_queries) >= 6:
                break

    query_candidates: list[str] = []
    for candidate in _derive_symbol_candidates(query_text, limit=4):
        lower = candidate.lower()
        if lower in seen_names:
            continue
        seen_names.add(lower)
        query_candidates.append(candidate)

    if not suggested_queries:
        for candidate in query_candidates:
            suggested_names.append(candidate)
            suggested_queries.append(
                {
                    "name": candidate,
                    "kind": None,
                    "exact": False,
                    "source_path": None,
                    "source_role": "query_candidate",
                    "reason": "Technical name derived from the initial query.",
                }
            )

    return {
        "recommended": bool(suggested_queries),
        "suggested_names": suggested_names[:6],
        "suggested_queries": suggested_queries[:6],
        "focus_paths": focus_paths[:4],
        "query_candidates": query_candidates[:4],
    }


def _symbol_follow_up_reason(*, role: str, kind: Any) -> str:
    normalized_role = str(role or "supporting").strip().lower()
    normalized_kind = str(kind or "").strip().lower()
    if normalized_role == "entry_point":
        return "Confirm the entry symbol and the exact lines of the application surface."
    if normalized_role == "implementation":
        return "Open the main implementation symbol linked to the flow."
    if normalized_role == "contract":
        return "Check the contract or interface used in the flow."
    if normalized_kind in {"method", "function"}:
        return "Disambiguate the relevant executable point surfaced by the context."
    return "Check the most relevant technical symbol surfaced by the context."


def _classify_database_target(host: Optional[str]) -> tuple[str, str]:
    normalized = str(host or "").strip().lower().strip("[]")
    if not normalized:
        return ("unknown", "unknown")

    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return ("loopback", "local_or_docker_port_mapping")

    if normalized.endswith(".docker.internal") or normalized in {
        "host.docker.internal",
        "gateway.docker.internal",
        "postgres",
        "postgresql",
        "pgvector",
        "db",
    }:
        return ("docker_alias", "docker_network_alias")

    try:
        parsed_ip = ipaddress.ip_address(normalized)
    except ValueError:
        return ("remote_hostname", "remote_or_managed_postgres")

    if parsed_ip.is_loopback:
        return ("loopback", "local_or_docker_port_mapping")
    if parsed_ip.is_private:
        return ("remote_private", "remote_or_lan_postgres")
    return ("remote_public", "remote_or_managed_postgres")


def _database_unreachable_action(deployment_hint: str) -> str:
    if deployment_hint == "local_or_docker_port_mapping":
        return (
            "With a local Postgres check the service and port; with Docker port mapping "
            "check that the container is running and the DSN host port is published."
        )
    if deployment_hint == "docker_network_alias":
        return (
            "With Postgres in Docker without port mapping check the Docker network, the service/container "
            "hostname and reachability from the MCP environment."
        )
    if deployment_hint in {"remote_or_lan_postgres", "remote_or_managed_postgres"}:
        return (
            "With a remote or cloud Postgres check network reachability, firewall/VPN, sslmode "
            "and DSN credentials."
        )
    return "Check LLM_CONTEXT_DSN, runtime credentials and PostgreSQL reachability."


def _pgvector_missing_action(deployment_hint: str) -> str:
    if deployment_hint == "remote_or_managed_postgres":
        return (
            "Check that the remote PostgreSQL instance supports pgvector and install/enable "
            "the extension on the target database."
        )
    return "Install or enable the pgvector extension on the target database before using the runtime."


def _schema_missing_action(deployment_hint: str) -> str:
    if deployment_hint in {"remote_or_lan_postgres", "remote_or_managed_postgres"}:
        return (
            "Run init-db-v2 against the PostgreSQL target configured in the DSN before using the read-plane."
        )
    return "Run init-db-v2 on the target database before using the read-plane."


def _derive_symbol_candidates(query_text: Any, *, limit: int = 4) -> list[str]:
    if not isinstance(query_text, str):
        return []
    text = query_text.strip()
    if not text:
        return []
    candidates: list[str] = []
    parts = [part for part in re.split(r"[^A-Za-z0-9_]+", text) if part]
    if len(parts) == 1 and _looks_like_symbol(parts[0]):
        candidates.append(parts[0])
    for part in parts:
        if _looks_like_symbol(part):
            candidates.append(part)
        if len(candidates) >= limit:
            break
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        normalized = item.strip()
        lower = normalized.lower()
        if lower in seen:
            continue
        seen.add(lower)
        result.append(normalized)
    return result[:limit]


def _looks_like_symbol(token: str) -> bool:
    if len(token) < 3:
        return False
    if token.strip().lower() in {
        "apri",
        "open",
        "show",
        "mostra",
        "trova",
        "find",
        "cerca",
        "usa",
        "call",
        "chiama",
        "metodo",
        "method",
        "classe",
        "class",
        "interfaccia",
        "interface",
        "servizio",
        "service",
    }:
        return False
    if "_" in token:
        return True
    if any(char.isupper() for char in token[1:]):
        return True
    return token.isidentifier() and token.lower() != token

