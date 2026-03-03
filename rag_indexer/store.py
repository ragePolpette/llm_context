from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import uuid

import psycopg
from psycopg.types.json import Json

from .db import execute_many_params, execute_params


@dataclass
class DocumentRecord:
    repo_id: str
    path: str
    path_norm: str
    ext: str
    language: Optional[str]
    doc_type: str
    size_bytes: int
    mtime: Optional[datetime]
    content_hash: str
    is_generated: bool = False
    confidentiality: str = "internal"
    tags: dict[str, Any] = field(default_factory=dict)
    doc_id: uuid.UUID = field(default_factory=uuid.uuid4)


@dataclass
class ChunkRecord:
    doc_id: uuid.UUID
    chunk_index: int
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    text: str
    text_hash: str
    section_path: str = ""
    token_count: Optional[int] = None
    quality_flags: dict[str, Any] = field(default_factory=dict)
    is_header: bool = False
    chunk_id: uuid.UUID = field(default_factory=uuid.uuid4)


@dataclass
class EmbeddingRecord:
    chunk_id: uuid.UUID
    model_id: str
    dim: int
    embedding: list[float]
    embedding_id: uuid.UUID = field(default_factory=uuid.uuid4)


class RagStore:
    def __init__(self, conn: psycopg.Connection, embedding_dim: int) -> None:
        self.conn = conn
        self.embedding_dim = int(embedding_dim)

    def delete_repo(self, repo_id: str) -> int:
        sql_text = "DELETE FROM documents WHERE repo_id = @Valore0"
        with self.conn.cursor() as cur:
            execute_params(cur, sql_text, [repo_id])
            deleted = cur.rowcount or 0
        self.conn.commit()
        return deleted

    def get_document(self, repo_id: str, path_norm: str) -> Optional[dict[str, Any]]:
        sql_text = (
            "SELECT doc_id, content_hash, deleted_at "
            "FROM documents WHERE repo_id = @Valore0 AND path_norm = @Valore1"
        )
        with self.conn.cursor() as cur:
            execute_params(cur, sql_text, [repo_id, path_norm])
            row = cur.fetchone()
        if not row:
            return None
        return {"doc_id": row[0], "content_hash": row[1], "deleted_at": row[2]}

    def list_documents(self, repo_id: str) -> list[dict[str, Any]]:
        sql_text = (
            "SELECT doc_id, path_norm, content_hash, deleted_at "
            "FROM documents WHERE repo_id = @Valore0"
        )
        with self.conn.cursor() as cur:
            execute_params(cur, sql_text, [repo_id])
            rows = cur.fetchall()
        return [
            {
                "doc_id": row[0],
                "path_norm": row[1],
                "content_hash": row[2],
                "deleted_at": row[3],
            }
            for row in rows
        ]

    def upsert_document(self, record: DocumentRecord) -> uuid.UUID:
        sql_text = (
            "INSERT INTO documents ("
            "doc_id, repo_id, path, path_norm, ext, language, doc_type, size_bytes, "
            "mtime, content_hash, is_generated, confidentiality, tags, updated_at, "
            "last_indexed_at, deleted_at"
            ") VALUES ("
            "@Valore0, @Valore1, @Valore2, @Valore3, @Valore4, @Valore5, "
            "@Valore6, @Valore7, @Valore8, @Valore9, @Valore10, @Valore11, "
            "@Valore12, now(), now(), NULL"
            ") ON CONFLICT (repo_id, path_norm) DO UPDATE SET "
            "path = EXCLUDED.path, "
            "ext = EXCLUDED.ext, "
            "language = EXCLUDED.language, "
            "doc_type = EXCLUDED.doc_type, "
            "size_bytes = EXCLUDED.size_bytes, "
            "mtime = EXCLUDED.mtime, "
            "content_hash = EXCLUDED.content_hash, "
            "is_generated = EXCLUDED.is_generated, "
            "confidentiality = EXCLUDED.confidentiality, "
            "tags = EXCLUDED.tags, "
            "updated_at = now(), "
            "last_indexed_at = now(), "
            "deleted_at = NULL "
            "RETURNING doc_id"
        )
        with self.conn.cursor() as cur:
            execute_params(
                cur,
                sql_text,
                [
                    record.doc_id,
                    record.repo_id,
                    record.path,
                    record.path_norm,
                    record.ext,
                    record.language,
                    record.doc_type,
                    record.size_bytes,
                    record.mtime,
                    record.content_hash,
                    record.is_generated,
                    record.confidentiality,
                    Json(record.tags or {}),
                ],
            )
            row = cur.fetchone()
        self.conn.commit()
        return row[0]

    def mark_documents_deleted(self, repo_id: str, doc_ids: list[uuid.UUID]) -> int:
        if not doc_ids:
            return 0
        sql_text = (
            "UPDATE documents SET deleted_at = now() "
            "WHERE repo_id = @Valore0 AND doc_id = ANY(@Valore1)"
        )
        with self.conn.cursor() as cur:
            execute_params(cur, sql_text, [repo_id, doc_ids])
            updated = cur.rowcount or 0
        self.conn.commit()
        return updated

    def delete_chunks(self, doc_id: uuid.UUID) -> int:
        sql_text = "DELETE FROM chunks WHERE doc_id = @Valore0"
        with self.conn.cursor() as cur:
            execute_params(cur, sql_text, [doc_id])
            deleted = cur.rowcount or 0
        self.conn.commit()
        return deleted

    def insert_chunks_and_embeddings(
        self,
        chunks: list[ChunkRecord],
        embeddings: list[EmbeddingRecord],
        batch_size: int = 200,
    ) -> int:
        if not chunks:
            return 0
        chunk_sql = (
            "INSERT INTO chunks ("
            "chunk_id, doc_id, chunk_index, char_start, char_end, line_start, line_end, "
            "section_path, text, text_hash, token_count, quality_flags, is_header, search_tsv"
            ") VALUES ("
            "@Valore0, @Valore1, @Valore2, @Valore3, @Valore4, @Valore5, @Valore6, "
            "@Valore7, @Valore8, @Valore9, @Valore10, @Valore11, @Valore12, "
            "to_tsvector('italian_unaccent', @Valore13)"
            ")"
        )
        embedding_sql = (
            "INSERT INTO chunk_embeddings ("
            "embedding_id, chunk_id, model_id, dim, embedding"
            ") VALUES ("
            "@Valore0, @Valore1, @Valore2, @Valore3, @Valore4"
            ")"
        )
        total = 0
        with self.conn.cursor() as cur:
            chunk_batch: list[list[Any]] = []
            for chunk in chunks:
                chunk_batch.append(
                    [
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.chunk_index,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.section_path,
                        chunk.text,
                        chunk.text_hash,
                        chunk.token_count,
                        Json(chunk.quality_flags or {}),
                        chunk.is_header,
                        chunk.text,
                    ]
                )
                if len(chunk_batch) >= batch_size:
                    execute_many_params(cur, chunk_sql, chunk_batch)
                    total += len(chunk_batch)
                    chunk_batch = []
            if chunk_batch:
                execute_many_params(cur, chunk_sql, chunk_batch)
                total += len(chunk_batch)

            embed_batch: list[list[Any]] = []
            for emb in embeddings:
                _validate_embedding(emb.embedding, self.embedding_dim)
                embed_batch.append(
                    [
                        emb.embedding_id,
                        emb.chunk_id,
                        emb.model_id,
                        emb.dim,
                        emb.embedding,
                    ]
                )
                if len(embed_batch) >= batch_size:
                    execute_many_params(cur, embedding_sql, embed_batch)
                    embed_batch = []
            if embed_batch:
                execute_many_params(cur, embedding_sql, embed_batch)
        self.conn.commit()
        return total

    def query_vector(
        self,
        embedding: list[float],
        repo_id: str,
        model_id: str,
        top_k: int,
        path_prefix: Optional[str] = None,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        _validate_embedding(embedding, self.embedding_dim)
        params: list[Any] = []

        def add_param(value: Any) -> str:
            placeholder = f"@Valore{len(params)}"
            params.append(value)
            return placeholder

        emb_placeholder = add_param(embedding)
        repo_placeholder = add_param(repo_id)
        model_placeholder = add_param(model_id)
        dim_placeholder = add_param(self.embedding_dim)

        conditions = [
            f"d.repo_id = {repo_placeholder}",
            "d.deleted_at IS NULL",
            f"e.model_id = {model_placeholder}",
            f"e.dim = {dim_placeholder}",
        ]

        if path_prefix:
            path_placeholder = add_param(f"{path_prefix}%")
            conditions.append(f"d.path_norm LIKE {path_placeholder}")
        if doc_type:
            doc_placeholder = add_param(doc_type)
            conditions.append(f"d.doc_type = {doc_placeholder}")
        if language:
            lang_placeholder = add_param(language)
            conditions.append(f"d.language = {lang_placeholder}")

        limit_placeholder = add_param(top_k)

        sql_text = (
            "SELECT c.chunk_id, d.doc_id, d.path, d.path_norm, c.chunk_index, "
            "c.text, c.text_hash, c.line_start, c.line_end, c.char_start, c.char_end, "
            "c.section_path, c.is_header, "
            f"(e.embedding <=> ({emb_placeholder})::vector) AS distance "
            "FROM chunk_embeddings e "
            "JOIN chunks c ON c.chunk_id = e.chunk_id "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE "
            + " AND ".join(conditions)
            + " ORDER BY distance "
            + f"LIMIT {limit_placeholder}"
        )
        results: list[dict[str, Any]] = []
        with self.conn.cursor() as cur:
            execute_params(cur, sql_text, params)
            rows = cur.fetchall()
        for row in rows:
            results.append(
                {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "source_path": row[2],
                    "path_norm": row[3],
                    "chunk_index": row[4],
                    "text": row[5],
                    "text_hash": row[6],
                    "line_start": row[7],
                    "line_end": row[8],
                    "char_start": row[9],
                    "char_end": row[10],
                    "section_path": row[11] or "",
                    "is_header": bool(row[12]),
                    "distance": float(row[13]),
                }
            )
        return results

    def query_keyword(
        self,
        query_text: str,
        repo_id: str,
        top_k: int,
        path_prefix: Optional[str] = None,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []

        def add_param(value: Any) -> str:
            placeholder = f"@Valore{len(params)}"
            params.append(value)
            return placeholder

        query_placeholder = add_param(query_text)
        repo_placeholder = add_param(repo_id)
        rank_placeholder = add_param(query_text)

        conditions = [
            f"d.repo_id = {repo_placeholder}",
            "d.deleted_at IS NULL",
            f"c.search_tsv @@ plainto_tsquery('italian_unaccent', {query_placeholder})",
        ]

        if path_prefix:
            path_placeholder = add_param(f"{path_prefix}%")
            conditions.append(f"d.path_norm LIKE {path_placeholder}")
        if doc_type:
            doc_placeholder = add_param(doc_type)
            conditions.append(f"d.doc_type = {doc_placeholder}")
        if language:
            lang_placeholder = add_param(language)
            conditions.append(f"d.language = {lang_placeholder}")

        limit_placeholder = add_param(top_k)

        sql_text = (
            "SELECT c.chunk_id, d.doc_id, d.path, d.path_norm, c.chunk_index, "
            "c.text, c.text_hash, c.line_start, c.line_end, c.char_start, c.char_end, "
            "c.section_path, c.is_header, "
            f"ts_rank_cd(c.search_tsv, plainto_tsquery('italian_unaccent', {rank_placeholder})) AS rank "
            "FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE "
            + " AND ".join(conditions)
            + " ORDER BY rank DESC "
            + f"LIMIT {limit_placeholder}"
        )
        results: list[dict[str, Any]] = []
        with self.conn.cursor() as cur:
            execute_params(cur, sql_text, params)
            rows = cur.fetchall()
        for row in rows:
            results.append(
                {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "source_path": row[2],
                    "path_norm": row[3],
                    "chunk_index": row[4],
                    "text": row[5],
                    "text_hash": row[6],
                    "line_start": row[7],
                    "line_end": row[8],
                    "char_start": row[9],
                    "char_end": row[10],
                    "section_path": row[11] or "",
                    "is_header": bool(row[12]),
                    "rank": float(row[13]),
                }
            )
        return results


def _validate_embedding(embedding: list[float], dim: int) -> None:
    if len(embedding) != dim:
        raise ValueError(f"Embedding dim mismatch: {len(embedding)} != {dim}")
    for value in embedding:
        if not isinstance(value, (int, float)):
            raise TypeError("Embedding values must be numeric")
