"""
MCP Handler for RAG context/search tools.

Shared logic for both stdio and HTTP MCP servers.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from rag_indexer.agent_context import build_context
from rag_indexer.config import load_config
from rag_indexer.db import get_connection
from rag_indexer.embedder import (
    DummyEmbedder,
    Embedder,
    LocalHashEmbedder,
    LocalSentenceTransformerEmbedder,
)
from rag_indexer.path_utils import parse_bool, resolve_path_prefix
from rag_indexer.store import RagStore


def load_dotenv() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        import dotenv
    except Exception:
        return
    loader = getattr(dotenv, "load_dotenv", None)
    if callable(loader):
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
        self._embedder_lock = threading.Lock()
        self._default_dsn = self._require_default_dsn()
        self._default_project_id = os.getenv("LLM_CONTEXT_PROJECT_ID", "myproj")
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
                        tool_context_info(),
                        tool_symbol_search(),
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
                elif name == "context_info":
                    payload = self._run_context_info()
                elif name == "symbol_search":
                    payload = self._run_symbol_search(args)
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

    def _run_rag_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute rag_context tool."""
        query_text = args.get("query_text")
        query_embedding = args.get("query_embedding")
        if query_text is None and query_embedding is None:
            raise ValueError("query_text or query_embedding is required")
        project_id = args.get("project_id") or self._default_project_id
        top_k = int(args.get("top_k", 8))
        path_prefix = args.get("path_prefix")
        file_path = args.get("file")
        auto_scope = parse_bool(str(args.get("auto_scope", True)))
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
        context_sheet = format_context_sheet(
            query_text=query_text,
            path_prefix=resolved_prefix,
            top_k=top_k,
            results=results,
            max_chars=max_chars,
        )
        return {
            "context": context,
            "context_sheet": context_sheet,
            "results": results,
            "meta": {
                "project_id": project_id,
                "top_k": top_k,
                "path_prefix": resolved_prefix,
                "max_chars": max_chars,
            },
        }

    def _run_rag_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute rag_search tool (same as rag_context but without formatted context)."""
        payload = self._run_rag_context(args)
        payload.pop("context", None)
        return payload

    def _run_context_info(self) -> dict[str, Any]:
        """Describe scope and boundaries of llm-context MCP."""
        return {
            "server": "llm-context-mcp",
            "purpose": "Recupero contesto da codice/documenti indicizzati (RAG).",
            "capabilities": [
                "rag_context: contesto formattato per analisi codice/documenti",
                "rag_search: risultati raw di retrieval semantico/keyword",
                "symbol_search: lookup esatto/prefisso simboli per nome (class/function/method/...)",
            ],
            "boundaries": [
                "NON e' un sistema di memoria operativa persistente",
                "NON sostituisce llm-memory per decisioni/preferenze operative",
            ],
        }

    def _run_symbol_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute symbol_search tool."""
        name = args.get("name")
        if not name:
            raise ValueError("'name' is required for symbol_search")
        project_id = args.get("project_id") or self._default_project_id
        kind = args.get("kind") or None
        language = args.get("language") or None
        exact = parse_bool(str(args.get("exact", False)))
        limit = int(args.get("limit", 20))
        embedding_dim = int(args.get("embedding_dim", self._default_embedding_dim))

        conn = get_connection(self._default_dsn)
        try:
            store = RagStore(conn, embedding_dim)
            results = store.query_symbols(
                name=name,
                repo_id=project_id,
                kind=kind,
                language=language,
                exact=exact,
                limit=limit,
            )
        finally:
            conn.close()

        return {
            "query": {"name": name, "kind": kind, "language": language, "exact": exact},
            "count": len(results),
            "results": results,
        }

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
            "Recupera contesto RAG da codice/documenti indicizzati. "
            "Usare solo per context retrieval tecnico; non salva memorie operative."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "query_embedding": {"type": "array", "items": {"type": "number"}},
                "project_id": {"type": "string"},
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
                    "description": "text (default), sheet, or json",
                },
            },
        },
    }


def tool_rag_search() -> dict[str, Any]:
    """Return the rag_search tool schema."""
    tool = tool_rag_context()
    tool["name"] = "rag_search"
    tool["description"] = (
        "Recupera match RAG raw su codice/documenti (no contesto formattato). "
        "Non e' un memory store operativo."
    )
    return tool


def tool_context_info() -> dict[str, Any]:
    """Return tool schema describing llm-context purpose and boundaries."""
    return {
        "name": "context_info",
        "description": "Spiega scopo/limiti del MCP llm-context.",
        "inputSchema": {"type": "object", "properties": {}},
    }


def tool_symbol_search() -> dict[str, Any]:
    """Return the symbol_search tool schema."""
    return {
        "name": "symbol_search",
        "description": (
            "Cerca simboli (class, function, method, interface, enum, struct, type) per nome "
            "nel codice indicizzato. Restituisce line_start, line_end, signature e path del file."
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
                "project_id": {"type": "string"},
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
        if format_hint in {"sheet", "context-sheet"}:
            return str(payload.get("context_sheet", ""))
        return str(payload.get("context", ""))
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
