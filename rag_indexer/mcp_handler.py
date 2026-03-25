"""
MCP Handler for RAG context/search tools.

Shared logic for both stdio and HTTP MCP servers.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rag_indexer.agent_context import build_context
from rag_indexer.config import load_config
from rag_indexer.context_assembler import assemble_functional_context
from rag_indexer.db import get_connection, get_pool
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


def load_dotenv() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        import dotenv
    except Exception:
        return
    loader = getattr(dotenv, "load_dotenv", None)
    if callable(loader):
        dotenv_path = str(os.getenv("LLM_CONTEXT_DOTENV_PATH", "")).strip()
        if dotenv_path:
            loader(dotenv_path=dotenv_path)
            return
        loader()


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles UUID objects."""

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


class MCPHandler:
    """Handles MCP tool calls (shared logic between stdio and HTTP servers)."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        load_dotenv()
        self._project_root = Path(__file__).resolve().parent.parent
        self._configure_local_model_dirs()
        if config_path is None:
            config_path = str(self._project_root / "config.yaml")
        self._config = load_config(config_path)
        self._project_registry = load_project_registry(
            self._config.projects_registry_path,
            self._config.projects_state_path,
        )
        self._embedder_lock = threading.Lock()
        self._default_dsn = self._require_default_dsn()
        self._default_project_id = str(
            os.getenv("LLM_CONTEXT_PROJECT_ID", self._config.default_project_id)
        ).strip()
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
        self._max_query_embedding_items = int(
            os.getenv("LLM_CONTEXT_MAX_QUERY_EMBEDDING_ITEMS", "4096")
        )
        self._embedder_cache: dict[str, Embedder] = {}
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
            args = {
                "embedder": self._default_embedder,
                "local_model": self._default_local_model,
            }
            self._get_embedder(args, self._default_embedding_dim)
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
                    "serverInfo": {"name": "llm-context-mcp", "version": "0.2.0"},
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
                        tool_map_work_item_to_codebase(),
                        tool_context_info(),
                        tool_symbol_search(),
                        tool_list_projects(),
                        tool_get_project_info(),
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

    def _run_rag_context(self, args: dict[str, Any], *, tool_name: str = "rag_context") -> dict[str, Any]:
        """Execute rag_context tool."""
        query_text = args.get("query_text")
        query_embedding = args.get("query_embedding")
        if query_text is None and query_embedding is None:
            raise ValueError("query_text or query_embedding is required")
        project_id = self._resolve_project_id(args, tool_name=tool_name)
        top_k = int(args.get("top_k", 8))
        path_prefix = args.get("path_prefix")
        file_path = args.get("file")
        auto_scope = parse_bool(str(args.get("auto_scope", True)))
        explicit_scope = bool(path_prefix or file_path)
        default_max_chars = int(os.getenv("LLM_CONTEXT_MAX_CHARS", "12000"))
        max_chars = int(args.get("max_chars", default_max_chars))
        doc_type = args.get("doc_type") or self._config.default_doc_type
        language = args.get("language")
        embedding_dim = int(args.get("embedding_dim", self._default_embedding_dim))
        self._validate_query_embedding(query_embedding, embedding_dim)
        embedder = self._get_embedder(args, embedding_dim)

        resolved_prefix = resolve_path_prefix(
            path_prefix=path_prefix,
            file_path=file_path,
            query_text=query_text,
            auto_scope=auto_scope,
            scope_map=self._config.scope_map,
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

        context, results = build_context(
            dsn=self._default_dsn,
            embedder=embedder,
            embedding_dim=embedding_dim,
            project_id=project_id,
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            path_prefix=resolved_prefix,
            max_chars=max_chars,
            doc_type=doc_type,
            language=language,
            vector_weight=self._config.vector_weight,
            keyword_weight=self._config.keyword_weight,
            max_chunks_per_doc=self._config.max_chunks_per_doc,
            min_score=self._config.min_score,
            header_penalty=self._config.header_penalty,
        )
        auto_scope_fallback_used = False
        initial_resolved_prefix = resolved_prefix
        if not results and resolved_prefix and auto_scope and not explicit_scope:
            context, results = build_context(
                dsn=self._default_dsn,
                embedder=embedder,
                embedding_dim=embedding_dim,
                project_id=project_id,
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                path_prefix=None,
                max_chars=max_chars,
                doc_type=doc_type,
                language=language,
                vector_weight=self._config.vector_weight,
                keyword_weight=self._config.keyword_weight,
                max_chunks_per_doc=self._config.max_chunks_per_doc,
                min_score=self._config.min_score,
                header_penalty=self._config.header_penalty,
            )
            resolved_prefix = None
            auto_scope_fallback_used = True
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
        payload = {
            "functional_context": functional_context,
            "context": context,
            "context_sheet": context_sheet,
            "results": results,
            "symbol_results": symbol_results,
            "meta": {
                "project_id": project_id,
                "top_k": top_k,
                "path_prefix": resolved_prefix,
                "max_chars": max_chars,
                "auto_scope_fallback_used": auto_scope_fallback_used,
                "initial_path_prefix": initial_resolved_prefix,
            },
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
        return payload

    def _run_rag_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute rag_search tool returning raw retrieval data."""
        payload = self._run_rag_context(args, tool_name="rag_search")
        payload.pop("functional_context", None)
        payload.pop("context", None)
        payload.pop("context_sheet", None)
        payload.pop("symbol_results", None)
        return payload

    def _run_context_info(self) -> dict[str, Any]:
        """Describe scope and boundaries of llm-context MCP."""
        return {
            "server": "llm-context-mcp",
            "purpose": "Recupero contesto da codice/documenti indicizzati (RAG).",
            "multi_project_enabled": self._config.multi_project_enabled,
            "write_enabled": self._config.write_enabled,
            "ingest_enabled": self._config.ingest_enabled,
            "project_count": self._project_registry.project_count(),
            "default_project_id": self._default_project_id,
            "tool_map": {
                "working_context": ["rag_context", "map_work_item_to_codebase"],
                "inspection": ["rag_search", "symbol_search"],
                "discovery": ["context_info", "list_projects", "get_project_info"],
            },
            "tool_roles": {
                "rag_context": {
                    "role": "default working tool",
                    "use_when": [
                        "devi ottenere velocemente il pacchetto di contesto piu' utile per lavorare sul codice",
                        "vuoi entry point, file core e contesto assemblato invece dei soli hit raw",
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
                        "vuoi vedere i match raw del retrieval",
                        "devi approfondire, confermare o debuggare la copertura della query",
                    ],
                    "returns": ["results", "meta"],
                },
                "symbol_search": {
                    "role": "precision lookup",
                    "use_when": [
                        "devi trovare signature e linee esatte di un simbolo",
                        "devi disambiguare nomi tecnici emersi da rag_context",
                    ],
                    "returns": ["results", "count"],
                },
                "map_work_item_to_codebase": {
                    "role": "functional-to-codebase mapping",
                    "use_when": [
                        "parti da ticket, work item o richiesta funzionale",
                        "vuoi repo_target, area, path e hint implementativo strutturati",
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
            "capabilities": [
                "rag_context: contesto formattato per analisi codice/documenti",
                "rag_search: risultati raw di retrieval semantico/keyword",
                "map_work_item_to_codebase: mapping strutturato tra richiesta funzionale e codebase",
                "symbol_search: lookup esatto/prefisso simboli per nome (class/function/method/...)",
                "list_projects: elenco dei progetti registrati",
                "get_project_info: dettaglio di un progetto registrato",
            ],
            "usage_notes": {
                "rag_context": (
                    "Tool principale per lavorare: restituisce un package funzionale assemblato di default."
                ),
                "rag_search": (
                    "Tool di approfondimento/raw: restituisce retrieval grezzo senza assembly."
                ),
                "map_work_item_to_codebase": (
                    "Tool strutturato per mappare richieste funzionali o ticket verso prodotto/repo/area."
                ),
                "symbol_search": (
                    "Tool di precisione per lookup simboli; utile per disambiguare nomi tecnici ed entry point."
                ),
            },
            "recommended_workflows": [
                {
                    "goal": "iniziare a lavorare su una query tecnica",
                    "steps": ["context_info", "rag_context", "symbol_search (se servono linee/signature esatte)"],
                },
                {
                    "goal": "approfondire o debuggare il retrieval",
                    "steps": ["rag_context", "rag_search", "symbol_search"],
                },
                {
                    "goal": "partire da ticket o richiesta funzionale",
                    "steps": ["map_work_item_to_codebase", "rag_context", "symbol_search"],
                },
            ],
            "boundaries": [
                "NON e' un sistema di memoria operativa persistente",
                "NON sostituisce llm-memory per decisioni/preferenze operative",
                "In multi-project mode le query richiedono project_id esplicito",
                "L'ingest non e' esposto come tool MCP standard",
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
        limit = int(args.get("limit", 20))
        embedding_dim = int(args.get("embedding_dim", self._default_embedding_dim))
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
            "top_k": int(args.get("top_k", 8)),
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
        projects = [project.to_public_dict() for project in self._project_registry.list_projects()]
        return {
            "count": len(projects),
            "multi_project_enabled": self._config.multi_project_enabled,
            "write_enabled": self._config.write_enabled,
            "ingest_enabled": self._config.ingest_enabled,
            "projects": projects,
        }

    def _run_get_project_info(self, args: dict[str, Any]) -> dict[str, Any]:
        project_id = str(args.get("project_id") or "").strip()
        if not project_id:
            raise ValueError("'project_id' is required for get_project_info")
        project = self._project_registry.require_project(project_id)
        return project.to_public_dict()

    def get_operational_status(self, *, ready: bool) -> dict[str, Any]:
        projects = []
        for project in self._project_registry.list_projects():
            projects.append(
                {
                    "project_id": project.project_id,
                    "display_name": project.display_name,
                    "ingest_enabled": project.ingest_enabled,
                    "write_enabled": project.write_enabled,
                    "last_ingest_status": project.last_ingest_status,
                    "last_successful_ingest_at": project.last_successful_ingest_at,
                    "index_version": project.index_version,
                    "index_fingerprint": project.index_fingerprint,
                }
            )
        return {
            "status": "ready" if ready else "loading",
            "multi_project_enabled": self._config.multi_project_enabled,
            "ingest_enabled": self._config.ingest_enabled,
            "write_enabled": self._config.write_enabled,
            "project_count": len(projects),
            "projects": projects,
        }

    def _resolve_project_id(self, args: dict[str, Any], *, tool_name: str) -> str:
        explicit_project_id = str(args.get("project_id") or "").strip()
        if self._config.multi_project_enabled:
            if not explicit_project_id:
                raise ValueError(
                    f"{tool_name} requires explicit project_id in multi-project mode. This is a read-plane MCP tool; "
                    "when multi_project_enabled=true, you must pass project_id explicitly and "
                    "no implicit default project is used."
                )
            return self._validate_known_project(explicit_project_id)

        if explicit_project_id:
            return self._validate_known_project(explicit_project_id)

        if self._project_registry.project_count() == 0:
            return self._default_project_id

        if self._default_project_id and self._project_registry.get_project(self._default_project_id):
            return self._default_project_id

        projects = self._project_registry.list_projects()
        if len(projects) == 1:
            return projects[0].project_id

        raise ValueError(
            f"{tool_name} could not resolve a safe default project in single-project mode. "
            "Pass project_id explicitly or configure a default project for the read-plane."
        )

    def _resolve_mapping_project_id(self, args: dict[str, Any]) -> str:
        explicit_project_id = str(args.get("project_id") or "").strip()
        if explicit_project_id:
            return self._validate_known_project(explicit_project_id)

        workspace_root = str(args.get("workspace_root") or "").strip()
        if workspace_root and self._project_registry.project_count() > 0:
            try:
                resolved = Path(workspace_root).expanduser().resolve()
            except Exception:
                resolved = Path(workspace_root)
            matches = []
            for project in self._project_registry.list_projects():
                try:
                    project_root = Path(project.root_path).resolve()
                except Exception:
                    project_root = Path(project.root_path)
                if resolved == project_root or str(resolved).startswith(str(project_root)) or str(project_root).startswith(str(resolved)):
                    matches.append(project.project_id)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    "workspace_root matches multiple registered projects. Pass project_id explicitly."
                )

        return self._resolve_project_id(args, tool_name="map_work_item_to_codebase")

    def _validate_known_project(self, project_id: str) -> str:
        if self._project_registry.project_count() == 0:
            return project_id
        return self._project_registry.require_project(project_id).project_id

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

    def _get_embedder(self, args: dict[str, Any], embedding_dim: int) -> Embedder:
        """Get or create embedder with caching."""
        name = str(args.get("embedder", self._default_embedder)).strip().lower()
        local_model = args.get("local_model") or self._default_local_model
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


def tool_rag_context() -> dict[str, Any]:
    """Return the rag_context tool schema."""
    return {
        "name": "rag_context",
        "description": (
            "Tool principale del read-plane MCP: restituisce un pacchetto di contesto funzionale "
            "assemblato da codice/documenti indicizzati, pensato per aiutare un agente a lavorare "
            "subito sul codice. Include entry point, core files, supporting matches, contesto "
            "assemblato e hint sui tool di follow-up. Usare solo per context retrieval tecnico; "
            "non salva memorie operative e non esegue ingest/index refresh. In single-project mode "
            "puo' usare il default project se configurato in modo sicuro; in multi-project mode "
            "richiede project_id esplicito e non usa alcun default implicito. Per "
            "write_enabled/ingest_enabled, refresh indice e operazioni di ingest-plane usare "
            "context_info e la CLI operativa."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "query_embedding": {"type": "array", "items": {"type": "number"}},
                "project_id": {
                    "type": "string",
                    "description": (
                        "Project scope per la retrieval. In multi-project mode "
                        "(multi_project_enabled=true) e' obbligatorio; in single-project mode "
                        "puo' essere omesso solo se il server supporta un default project sicuro."
                    ),
                },
                "top_k": {"type": "integer"},
                "path_prefix": {"type": "string"},
                "file": {"type": "string"},
                "auto_scope": {"type": "boolean"},
                "max_chars": {"type": "integer"},
                "embedder": {"type": "string"},
                "embedding_dim": {"type": "integer"},
                "local_model": {"type": "string"},
                "doc_type": {"type": "string"},
                "language": {"type": "string"},
                "format": {
                    "type": "string",
                    "description": "functional (default), legacy, sheet, or json",
                },
            },
        },
    }


def tool_rag_search() -> dict[str, Any]:
    """Return the rag_search tool schema."""
    tool = tool_rag_context()
    tool["name"] = "rag_search"
    tool["description"] = (
        "Tool di approfondimento del read-plane MCP: restituisce match RAG raw su "
        "codice/documenti indicizzati senza context assembly. Usarlo per ricerca mirata, "
        "debug del retrieval, conferme puntuali o quando serve vedere i risultati grezzi oltre "
        "il pacchetto principale di rag_context. Non e' un memory store operativo e non esegue "
        "ingest o refresh indice. In single-project mode puo' usare un default project solo se "
        "supportato in modo sicuro; in multi-project mode richiede project_id esplicito."
    )
    return tool


def tool_map_work_item_to_codebase() -> dict[str, Any]:
    return {
        "name": "map_work_item_to_codebase",
        "description": (
            "Mappa una richiesta funzionale o ticket verso la codebase indicizzata e restituisce "
            "un payload strutturato con repo_target, area, fattibilita', hint implementativo, path "
            "rilevanti e blocker. Usa il motore di contesto funzionale del read-plane."
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
                    "description": "Project scope esplicito. Preferito quando disponibile.",
                },
                "workspace_root": {
                    "type": "string",
                    "description": "Workspace o repo root locale da usare per risolvere il project scope.",
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
            "Tool di discovery del server: espone scopo, limiti, tool disponibili, ruoli "
            "consigliati e workflow d'uso del MCP llm-context."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    }


def tool_list_projects() -> dict[str, Any]:
    return {
        "name": "list_projects",
        "description": "Elenca i progetti registrati e il relativo stato operativo di ingest/index.",
        "inputSchema": {"type": "object", "properties": {}},
    }


def tool_get_project_info() -> dict[str, Any]:
    return {
        "name": "get_project_info",
        "description": "Restituisce il dettaglio operativo di un progetto registrato.",
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
            "Cerca simboli (class, function, method, interface, enum, struct, type) per nome "
            "nel codice indicizzato. E' il tool di precisione da usare dopo rag_context quando "
            "serve disambiguare nomi tecnici, trovare signature/linee esatte o confermare "
            "entry point specifici. E' un tool di read-plane: non aggiorna l'indice e non fa "
            "operazioni di ingest-plane. In multi-project mode richiede project_id esplicito; in "
            "single-project mode puo' usare il default project solo se configurato in modo "
            "sicuro. Restituisce line_start, line_end, signature e path del file."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome (o prefisso) del simbolo da cercare",
                },
                "kind": {
                    "type": "string",
                    "description": "Filtro tipo: class | function | method | interface | enum | struct | type",
                },
                "project_id": {
                    "type": "string",
                    "description": (
                        "Project scope per la ricerca simboli. In multi-project mode "
                        "(multi_project_enabled=true) e' obbligatorio; in single-project mode "
                        "puo' essere omesso solo se il server espone un default project sicuro."
                    ),
                },
                "language": {
                    "type": "string",
                    "description": "Filtro linguaggio: python | javascript | typescript | csharp",
                },
                "exact": {
                    "type": "boolean",
                    "description": "Se true: match esatto (case-insensitive); se false (default): match prefisso",
                },
                "limit": {"type": "integer"},
                "embedding_dim": {"type": "integer"},
            },
        },
    }


def format_tool_text(name: str, args: dict[str, Any], payload: dict[str, Any]) -> str:
    """Format tool output based on format hint."""
    format_hint = str(args.get("format", "") or "").strip().lower()
    if name == "rag_context":
        if format_hint in {"json", "full"}:
            return json.dumps(payload, indent=2, ensure_ascii=True, cls=UUIDEncoder)
        if format_hint in {"legacy", "text"}:
            return str(payload.get("context", ""))
        if format_hint in {"sheet", "context-sheet"}:
            return str(payload.get("context_sheet", ""))
        return format_functional_context_text(payload.get("functional_context") or {})
    return json.dumps(payload, indent=2, ensure_ascii=True, cls=UUIDEncoder)


def format_context_sheet(
    query_text: Optional[str],
    path_prefix: Optional[str],
    top_k: int,
    results: list[dict[str, Any]],
    max_chars: int,
) -> str:
    """Format results as a structured context sheet."""
    safe_query = (query_text or "").strip() or "(vuoto)"
    lines: list[str] = [
        "CONTEXT",
        f"query: {safe_query}",
        f"scope: {path_prefix or '(nessuno)'}",
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
        f"query: {str(query.get('text') or '(vuoto)')}",
        f"scope: {str(query.get('path_prefix') or '(nessuno)')}",
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


def _build_tool_hints(
    *,
    query_text: Any,
    functional_context: dict[str, Any],
    results: list[dict[str, Any]],
    symbol_results: list[dict[str, Any]],
) -> dict[str, Any]:
    entry_points = functional_context.get("entry_points") or []
    symbol_names: list[str] = []
    seen_names: set[str] = set()
    for item in entry_points:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        lower = name.lower()
        if lower in seen_names:
            continue
        seen_names.add(lower)
        symbol_names.append(name)
        if len(symbol_names) >= 4:
            break

    if not symbol_names:
        for item in symbol_results:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            lower = name.lower()
            if lower in seen_names:
                continue
            seen_names.add(lower)
            symbol_names.append(name)
            if len(symbol_names) >= 4:
                break

    query_candidates = [
        candidate
        for candidate in _derive_symbol_candidates(query_text, limit=4)
        if candidate.lower() not in seen_names
    ]

    symbol_reason = (
        "Usa symbol_search per confermare signature, linee esatte ed entry point rilevanti."
        if symbol_names or symbol_results
        else "Usa symbol_search se devi disambiguare nomi tecnici o verificare un simbolo preciso."
    )
    raw_reason = (
        "Usa rag_search per vedere i match raw, approfondire varianti della query o investigare "
        "oltre il pacchetto principale."
        if results
        else "Usa rag_search con query piu' larghe quando rag_context non restituisce abbastanza segnali."
    )

    return {
        "primary_tool": "rag_context",
        "recommended_follow_up": [
            {
                "tool": "symbol_search",
                "reason": symbol_reason,
                "suggested_names": symbol_names or query_candidates,
            },
            {
                "tool": "rag_search",
                "reason": raw_reason,
            },
            {
                "tool": "context_info",
                "reason": "Usa context_info per rileggere ruoli, limiti e workflow consigliati dei tool MCP.",
            },
        ],
    }


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
    if "_" in token:
        return True
    if any(char.isupper() for char in token[1:]):
        return True
    return token.isidentifier() and token.lower() != token
