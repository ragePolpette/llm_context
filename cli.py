from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from rag_indexer.config import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_INCLUDE_EXTENSIONS,
    load_config,
    parse_csv_list,
)
from rag_indexer.db import get_connection, init_db_v2
from rag_indexer.embedder import (
    DummyEmbedder,
    Embedder,
    LocalHashEmbedder,
    LocalSentenceTransformerEmbedder,
)
from rag_indexer.agent_context import build_context
from rag_indexer.ingest import ingest_project_v2
from rag_indexer.retrieval import retrieve_v2
from rag_indexer.store import RagStore
from rag_indexer.path_utils import (
    normalize_path_prefix,
    resolve_path_prefix,
    parse_bool,
)


def _load_dotenv() -> None:
    try:
        dotenv = importlib.import_module("dotenv")
    except ImportError:
        return
    loader = getattr(dotenv, "load_dotenv", None)
    if callable(loader):
        loader()


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="RAG Memory Indexer")
    parser.add_argument("--config", default=None)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_v2_parser = subparsers.add_parser("init-db-v2", help="Init DB schema (v2)")
    init_v2_parser.add_argument("--dsn", required=True)
    init_v2_parser.add_argument("--embedding-dim", type=int, default=None)
    init_v2_parser.add_argument("--ivfflat-lists", type=int, default=None)

    ingest_parser = subparsers.add_parser("ingest", help="Full reindex")
    ingest_parser.add_argument("--dsn", required=True)
    ingest_parser.add_argument("--project-id", required=True)
    ingest_parser.add_argument("--root", required=True)
    ingest_parser.add_argument("--include", default=None)
    ingest_parser.add_argument("--exclude", default=None)
    ingest_parser.add_argument("--include-dirs", default=None)
    ingest_parser.add_argument("--assets-template-only", default=None)
    ingest_parser.add_argument("--assets-root-dir", default=None)
    ingest_parser.add_argument("--assets-template-dir", default=None)
    ingest_parser.add_argument("--max-bytes", type=int, default=None)
    ingest_parser.add_argument("--embedding-dim", type=int, default=None)
    ingest_parser.add_argument("--embedder", default="dummy")
    ingest_parser.add_argument("--gemini-api-key", default=None)
    ingest_parser.add_argument("--gemini-model", default="text-embedding-004")
    ingest_parser.add_argument(
        "--local-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    ingest_parser.add_argument("--chunk-size", type=int, default=None)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=None)
    ingest_parser.add_argument("--md-chunk-size", type=int, default=None)
    ingest_parser.add_argument("--code-chunk-size", type=int, default=None)
    ingest_parser.add_argument("--min-chunk-chars", type=int, default=None)
    ingest_parser.add_argument("--incremental", action="store_true")

    query_parser = subparsers.add_parser("query", help="Vector search")
    query_parser.add_argument("--dsn", required=True)
    query_parser.add_argument("--project-id", required=True)
    query_parser.add_argument("--text", default=None)
    query_parser.add_argument("--embedding", default=None)
    query_parser.add_argument("--top-k", type=int, default=8)
    query_parser.add_argument("--path-prefix", default=None)
    query_parser.add_argument("--file", default=None)
    query_parser.add_argument("--auto-scope", default="true")
    query_parser.add_argument("--embedding-dim", type=int, default=None)
    query_parser.add_argument("--embedder", default="dummy")
    query_parser.add_argument("--gemini-api-key", default=None)
    query_parser.add_argument("--gemini-model", default="text-embedding-004")
    query_parser.add_argument(
        "--local-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    query_parser.add_argument("--explain", action="store_true")
    query_parser.add_argument("--doc-type", default=None)
    query_parser.add_argument("--language", default=None)

    context_parser = subparsers.add_parser("context", help="Get context for agents")
    context_parser.add_argument("--dsn", required=True)
    context_parser.add_argument("--project-id", required=True)
    context_parser.add_argument("--text", default=None)
    context_parser.add_argument("--embedding", default=None)
    context_parser.add_argument("--top-k", type=int, default=8)
    context_parser.add_argument("--path-prefix", default=None)
    context_parser.add_argument("--file", default=None)
    context_parser.add_argument("--auto-scope", default="true")
    context_parser.add_argument("--stdin", action="store_true")
    context_parser.add_argument("--embedding-dim", type=int, default=None)
    context_parser.add_argument("--embedder", default="local-st")
    context_parser.add_argument("--gemini-api-key", default=None)
    context_parser.add_argument("--gemini-model", default="text-embedding-004")
    context_parser.add_argument(
        "--local-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    context_parser.add_argument("--max-chars", type=int, default=4000)
    context_parser.add_argument("--print-results", action="store_true")
    context_parser.add_argument("--doc-type", default=None)
    context_parser.add_argument("--language", default=None)

    args = parser.parse_args()
    _configure_logging(args.verbose)
    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)

    if args.command == "init-db-v2":
        embedding_dim = args.embedding_dim or config.embedding_dim
        ivfflat_lists = args.ivfflat_lists or config.ivfflat_lists
        conn = get_connection(args.dsn)
        init_db_v2(conn, embedding_dim, ivfflat_lists)
        print("DB initialized (v2)")
        return

    if args.command == "ingest":
        embedding_dim = args.embedding_dim or config.embedding_dim
        embedder = _build_embedder(
            args.embedder,
            embedding_dim,
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
            local_model=args.local_model,
        )
        conn = get_connection(args.dsn)
        include_patterns = parse_csv_list(args.include) or DEFAULT_INCLUDE_EXTENSIONS
        exclude_patterns = parse_csv_list(args.exclude) or DEFAULT_EXCLUDE_DIRS
        include_dirs = parse_csv_list(args.include_dirs) or config.include_dirs
        exclude_globs = config.exclude_globs
        never_index_ext = config.never_index_ext
        assets_template_only = (
            parse_bool(args.assets_template_only)
            if args.assets_template_only is not None
            else config.assets_template_only
        )
        assets_root_dir = args.assets_root_dir or config.assets_root_dir
        assets_template_dir = args.assets_template_dir or config.assets_template_dir
        max_bytes = args.max_bytes or config.max_bytes
        store = RagStore(conn, embedding_dim)
        stats = ingest_project_v2(
            store=store,
            embedder=embedder,
            repo_id=args.project_id,
            root=Path(args.root),
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            include_dirs=include_dirs,
            exclude_globs=exclude_globs,
            never_index_ext=never_index_ext,
            assets_template_only=assets_template_only,
            assets_root_dir=assets_root_dir,
            assets_template_dir=assets_template_dir,
            max_bytes=max_bytes,
            chunk_size=args.chunk_size or config.chunk_size,
            chunk_overlap=args.chunk_overlap or config.chunk_overlap,
            md_chunk_size=args.md_chunk_size or config.md_chunk_size,
            code_chunk_size=args.code_chunk_size or config.code_chunk_size,
            min_chunk_chars=args.min_chunk_chars or config.min_chunk_chars,
            incremental=args.incremental or config.incremental_ingest,
            logger=logging.getLogger("rag_indexer"),
        )
        print(
            "Ingest completed: "
            f"deleted={stats.deleted_rows} "
            f"files_scanned={stats.files_scanned} "
            f"files_indexed={stats.files_indexed} "
            f"files_skipped={stats.files_skipped} "
            f"chunks_created={stats.chunks_created} "
            f"chunks_inserted={stats.chunks_inserted} "
            f"duration_sec={stats.duration_sec:.2f}"
        )
        return

    if args.command == "query":
        embedding_dim = args.embedding_dim or config.embedding_dim
        embedder = _build_embedder(
            args.embedder,
            embedding_dim,
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
            local_model=args.local_model,
        )
        conn = get_connection(args.dsn)
        embedding = _parse_embedding(args.embedding)
        store = RagStore(conn, embedding_dim)
        doc_type = args.doc_type or config.default_doc_type
        results, explain = retrieve_v2(
            store=store,
            embedder=embedder,
            repo_id=args.project_id,
            top_k=args.top_k,
            text=args.text,
            embedding=embedding,
            path_prefix=resolve_path_prefix(
                args.path_prefix,
                args.file,
                args.text,
                parse_bool(args.auto_scope),
                config.scope_map,
            ),
            doc_type=doc_type,
            language=args.language,
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
            max_chunks_per_doc=config.max_chunks_per_doc,
            min_score=config.min_score,
            header_penalty=config.header_penalty,
        )
        for index, item in enumerate(results, start=1):
            print(
                f"{index}. score={item['score']:.4f} "
                f"path={item.get('source_path')} chunk={item.get('chunk_index')} "
                f"type={item.get('source_type', item.get('doc_type', ''))}\n{item['snippet']}"
            )
        if args.explain:
            print("Explain:")
            print(json.dumps(explain, indent=2, default=str))
        return

    if args.command == "context":
        embedding_dim = args.embedding_dim or config.embedding_dim
        text_input = _resolve_text(args.text, args.stdin)
        embedder = _build_embedder(
            args.embedder,
            embedding_dim,
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
            local_model=args.local_model,
        )
        embedding = _parse_embedding(args.embedding)
        doc_type = args.doc_type or config.default_doc_type
        context, results = build_context(
            dsn=args.dsn,
            embedder=embedder,
            embedding_dim=embedding_dim,
            project_id=args.project_id,
            query_text=text_input,
            query_embedding=embedding,
            top_k=args.top_k,
            path_prefix=resolve_path_prefix(
                args.path_prefix,
                args.file,
                text_input,
                parse_bool(args.auto_scope),
                config.scope_map,
            ),
            max_chars=args.max_chars,
            doc_type=doc_type,
            language=args.language,
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
            max_chunks_per_doc=config.max_chunks_per_doc,
            min_score=config.min_score,
            header_penalty=config.header_penalty,
        )
        print(context)
        if args.print_results:
            print("\n--- RESULTS ---")
            print(json.dumps(results, indent=2, default=str))
        return


def _build_embedder(
    name: str,
    dim: int,
    gemini_api_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
    local_model: Optional[str] = None,
) -> Embedder:
    normalized = name.strip().lower()
    if normalized == "local-hash":
        return LocalHashEmbedder(dim)
    if normalized in {"local-st", "sentence-transformers", "local"}:
        return LocalSentenceTransformerEmbedder(
            model_name=local_model
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    if normalized == "gemini":
        return GeminiEmbedder(
            api_key=gemini_api_key,
            model=gemini_model or "text-embedding-004",
        )
    if normalized == "dummy":
        return DummyEmbedder()
    raise ValueError(f"Unknown embedder: {name}")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def _parse_embedding(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError("Embedding must be a JSON list")
    return [float(item) for item in data]


def _resolve_text(text: Optional[str], use_stdin: bool) -> Optional[str]:
    if text is not None:
        return text
    if not use_stdin:
        return None
    data = sys.stdin.read()
    return data.strip() if data else None


def _resolve_config_path(config_arg: Optional[str]) -> Optional[str]:
    if config_arg:
        return config_arg
    default_path = Path(__file__).resolve().parent / "config.yaml"
    if default_path.exists():
        return str(default_path)
    return None


if __name__ == "__main__":
    main()
