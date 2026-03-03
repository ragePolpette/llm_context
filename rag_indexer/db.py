from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Any, Iterable, cast
import importlib
import re
import threading

_psycopg = importlib.import_module("psycopg")
sql = _psycopg.sql
register_vector = importlib.import_module("pgvector.psycopg").register_vector

# Connection pooling
_pools: dict[str, Any] = {}
_pool_lock = threading.Lock()


def get_pool(dsn: str, min_size: int = 2, max_size: int = 10) -> Any:
    """
    Get or create a connection pool for the given DSN.

    This is thread-safe and will reuse existing pools.
    """
    with _pool_lock:
        if dsn not in _pools:
            try:
                ConnectionPool = importlib.import_module("psycopg_pool").ConnectionPool
                _pools[dsn] = ConnectionPool(
                    dsn,
                    min_size=min_size,
                    max_size=max_size,
                    open=True,
                    configure=_configure_connection,
                )
            except ImportError:
                # Fallback: if psycopg_pool not available, return None
                # Caller should handle this gracefully
                return None
        return _pools[dsn]


def _configure_connection(conn: Any) -> None:
    """Configure a connection from the pool."""
    _register_vector_safe(conn)


def get_connection(dsn: str) -> Any:
    """
    Get a single connection (for CLI/batch operations).

    For server operations, prefer using get_pool() instead.
    """
    conn = _psycopg.connect(dsn)
    _register_vector_safe(conn)
    return conn


def init_db_v2(conn: Any, embedding_dim: int, ivfflat_lists: int) -> None:
    dim = _validate_positive_int(embedding_dim, "embedding_dim")
    lists = _validate_positive_int(ivfflat_lists, "ivfflat_lists")
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
        cur.execute("""
            DO $$
            BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'italian_unaccent') THEN
                CREATE TEXT SEARCH CONFIGURATION italian_unaccent (COPY = italian);
                ALTER TEXT SEARCH CONFIGURATION italian_unaccent
                  ALTER MAPPING FOR hword, hword_part, word
                  WITH unaccent, italian_stem;
              END IF;
            END
            $$;
        """)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id UUID PRIMARY KEY,
                repo_id TEXT NOT NULL,
                path TEXT NOT NULL,
                path_norm TEXT NOT NULL,
                ext TEXT NOT NULL,
                language TEXT NULL,
                doc_type TEXT NOT NULL,
                size_bytes INT NOT NULL,
                mtime TIMESTAMPTZ NULL,
                content_hash TEXT NOT NULL,
                is_generated BOOLEAN NOT NULL DEFAULT FALSE,
                confidentiality TEXT NOT NULL DEFAULT 'internal',
                tags JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_indexed_at TIMESTAMPTZ NULL,
                deleted_at TIMESTAMPTZ NULL,
                UNIQUE (repo_id, path_norm)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id UUID PRIMARY KEY,
                doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                char_start INT NOT NULL,
                char_end INT NOT NULL,
                line_start INT NOT NULL,
                line_end INT NOT NULL,
                section_path TEXT NULL,
                text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                token_count INT NULL,
                quality_flags JSONB NOT NULL DEFAULT '{}'::jsonb,
                is_header BOOLEAN NOT NULL DEFAULT FALSE,
                search_tsv tsvector,
                UNIQUE (doc_id, chunk_index)
            );
            """
        )
        table_sql = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                embedding_id UUID PRIMARY KEY,
                chunk_id UUID NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                dim INT NOT NULL,
                embedding vector({dim}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        ).format(dim=sql.Literal(dim))
        cur.execute(table_sql)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS index_runs (
                run_id UUID PRIMARY KEY,
                repo_id TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                dim INT NOT NULL,
                chunker_version TEXT NOT NULL,
                config_snapshot JSONB NOT NULL,
                started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                finished_at TIMESTAMPTZ NULL
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_repo_path_idx
            ON documents (repo_id, path_norm);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_repo_deleted_idx
            ON documents (repo_id, deleted_at);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_doc_idx
            ON chunks (doc_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_text_hash_idx
            ON chunks (text_hash);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_tsv_idx
            ON chunks USING GIN (search_tsv);
            """
        )
        ivfflat_sql = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS chunk_embeddings_ivfflat_idx
            ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists});
            """
        ).format(lists=sql.Literal(lists))
        cur.execute(ivfflat_sql)
    conn.commit()
    _register_vector_safe(conn)


def _register_vector_safe(conn: Any) -> None:
    try:
        register_vector(conn)
    except _psycopg.ProgrammingError:
        return


def execute_params(cursor: Any, sql_text: str, params: list[Any]) -> None:
    query, values = _normalize_params(sql_text, params)
    cursor.execute(sql.SQL(cast(Any, query)), values)


def execute_many_params(
    cursor: Any,
    sql_text: str,
    params_list: Iterable[list[Any]],
) -> None:
    query, _ = _normalize_params(sql_text, [])
    cursor.executemany(sql.SQL(cast(Any, query)), params_list)


def _normalize_params(sql_text: str, params: list[Any]) -> tuple[str, list[Any]]:
    query = re.sub(r"@Valore\d+", "%s", sql_text)
    return query, params


def _validate_positive_int(value: int, name: str) -> int:
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f"{name} must be > 0")
    return int_value
