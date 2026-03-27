from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Any, Iterable, cast
import importlib
import re
import threading

_psycopg = importlib.import_module("psycopg")
sql = _psycopg.sql
register_vector = importlib.import_module("pgvector.psycopg").register_vector
_PARAM_PATTERN = re.compile(r"@Valore(\d+)")

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


def inspect_database_runtime(
    dsn: str,
    *,
    connect_timeout: int = 3,
) -> dict[str, Any]:
    """Return a safe readiness summary for the configured PostgreSQL target."""
    timeout = _validate_positive_int(connect_timeout, "connect_timeout")
    result: dict[str, Any] = {
        "reachable": False,
        "database": None,
        "server_version": None,
        "pgvector_available": None,
        "schema_ready": None,
        "required_tables": {},
        "error": None,
    }
    conn = None
    try:
        conn = _psycopg.connect(dsn, connect_timeout=timeout)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT current_database(), current_setting('server_version', true);
                """
            )
            row = cur.fetchone() or (None, None)
            result["database"] = row[0]
            result["server_version"] = row[1]

            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_extension
                    WHERE extname = 'vector'
                );
                """
            )
            vector_row = cur.fetchone() or (False,)
            result["pgvector_available"] = bool(vector_row[0])

            cur.execute(
                """
                SELECT
                    to_regclass('public.documents') IS NOT NULL,
                    to_regclass('public.chunks') IS NOT NULL,
                    to_regclass('public.chunk_embeddings') IS NOT NULL,
                    to_regclass('public.index_runs') IS NOT NULL,
                    to_regclass('public.symbols') IS NOT NULL;
                """
            )
            tables_row = cur.fetchone() or (False, False, False, False, False)
            required_tables = {
                "documents": bool(tables_row[0]),
                "chunks": bool(tables_row[1]),
                "chunk_embeddings": bool(tables_row[2]),
                "index_runs": bool(tables_row[3]),
                "symbols": bool(tables_row[4]),
            }
            result["required_tables"] = required_tables
            result["schema_ready"] = all(required_tables.values())
            result["reachable"] = True
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if conn is not None:
            conn.close()


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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id  UUID PRIMARY KEY,
                doc_id     UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                repo_id    TEXT NOT NULL,
                name       TEXT NOT NULL,
                kind       TEXT NOT NULL,
                namespace  TEXT,
                line_start INT NOT NULL,
                line_end   INT,
                signature  TEXT,
                language   TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS symbols_name_lower_idx ON symbols (lower(name));
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS symbols_doc_idx ON symbols (doc_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS symbols_repo_kind_idx ON symbols (repo_id, kind);
            """
        )
    conn.commit()
    _register_vector_safe(conn)


def _register_vector_safe(conn: Any) -> None:
    try:
        register_vector(conn)
    except _psycopg.ProgrammingError:
        return


def execute_params(cursor: Any, sql_text: str, params: list[Any]) -> None:
    query, order = _normalize_params(sql_text)
    values = _reorder_params(order, params)
    cursor.execute(sql.SQL(cast(Any, query)), values)


def execute_many_params(
    cursor: Any,
    sql_text: str,
    params_list: Iterable[list[Any]],
) -> None:
    query, order = _normalize_params(sql_text)
    reordered_params = (_reorder_params(order, params) for params in params_list)
    cursor.executemany(sql.SQL(cast(Any, query)), reordered_params)


def _normalize_params(sql_text: str) -> tuple[str, list[int]]:
    indices: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        indices.append(int(match.group(1)))
        return "%s"

    query = _PARAM_PATTERN.sub(_replace, sql_text)
    _validate_placeholder_indices(indices, sql_text)
    return query, indices


def _validate_placeholder_indices(indices: list[int], sql_text: str) -> None:
    if not indices:
        return
    unique_indices = sorted(set(indices))
    expected = list(range(unique_indices[-1] + 1))
    if unique_indices != expected:
        raise ValueError(
            "SQL placeholders must be contiguous and start at @Valore0: "
            f"{sql_text}"
        )


def _reorder_params(order: list[int], params: list[Any]) -> list[Any]:
    if not order:
        if params:
            raise ValueError("SQL query does not define placeholders but parameters were provided")
        return []

    expected_param_count = max(order) + 1
    if len(params) != expected_param_count:
        raise ValueError(
            "SQL parameter count mismatch: "
            f"expected {expected_param_count}, got {len(params)}"
        )
    return [params[index] for index in order]


def _validate_positive_int(value: int, name: str) -> int:
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f"{name} must be > 0")
    return int_value
