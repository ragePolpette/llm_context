from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

from .chunking import Chunk, chunk_code, chunk_markdown, chunk_text
from .embedder import Embedder
from .scanner import get_source_type, read_text, scan_files
from .store import ChunkRecord, DocumentRecord, EmbeddingRecord, RagStore


@dataclass
class IngestStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    chunks_inserted: int = 0
    deleted_rows: int = 0
    duration_sec: float = 0.0


_DOC_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "llm_context.document")


def ingest_project_v2(
    store: RagStore,
    embedder: Embedder,
    repo_id: str,
    root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    include_dirs: list[str],
    exclude_globs: list[str],
    never_index_ext: list[str],
    assets_template_only: bool,
    assets_root_dir: str,
    assets_template_dir: str,
    max_bytes: int,
    chunk_size: int,
    chunk_overlap: int,
    md_chunk_size: int,
    code_chunk_size: int,
    min_chunk_chars: int,
    incremental: bool,
    logger: logging.Logger,
) -> IngestStats:
    stats = IngestStats()
    start_time = time.monotonic()
    if not incremental:
        stats.deleted_rows = store.delete_repo(repo_id)
    batch_size = _get_env_int("EMBED_BATCH_SIZE", 16)
    seen_paths: set[str] = set()
    model_id = _embedder_id(embedder)

    for path in scan_files(
        root,
        include_patterns,
        exclude_patterns,
        include_dirs=include_dirs,
        exclude_globs=exclude_globs,
        never_index_ext=never_index_ext,
        assets_template_only=assets_template_only,
        assets_root_dir=assets_root_dir,
        assets_template_dir=assets_template_dir,
    ):
        stats.files_scanned += 1
        relative_path = _relative_path(path, root)
        path_norm = _normalize_path(relative_path)
        seen_paths.add(path_norm)
        text, reason = read_text(path, max_bytes)
        if text is None:
            stats.files_skipped += 1
            logger.info("Skip file %s: %s", path, reason)
            continue
        normalized_text = _normalize_text(text)
        content_hash = _hash_text(normalized_text)
        existing = store.get_document(repo_id, path_norm)
        if (
            incremental
            and existing
            and existing.get("content_hash") == content_hash
            and existing.get("deleted_at") is None
        ):
            stats.files_skipped += 1
            continue

        source_type = get_source_type(path)
        doc_type = _guess_doc_type(source_type)
        language = _guess_language(path.suffix)
        file_stat = _safe_stat(path)
        mtime = _timestamp_or_none(file_stat)
        doc_id = uuid.uuid5(_DOC_NAMESPACE, f"{repo_id}:{path_norm}")

        size_bytes = file_stat.st_size if file_stat else len(
            text.encode("utf-8", errors="ignore")
        )
        doc_record = DocumentRecord(
            repo_id=repo_id,
            path=relative_path,
            path_norm=path_norm,
            ext=path.suffix.lower(),
            language=language,
            doc_type=doc_type,
            size_bytes=size_bytes,
            mtime=mtime,
            content_hash=content_hash,
            is_generated=False,
            confidentiality="internal",
            tags={},
            doc_id=doc_id,
        )

        doc_id = store.upsert_document(doc_record)
        store.delete_chunks(doc_id)

        if source_type == "md":
            chunks = chunk_markdown(text, md_chunk_size, chunk_overlap, min_chunk_chars)
        elif source_type == "code":
            chunks = chunk_code(text, code_chunk_size, min_chunk_chars)
            header_text = _build_code_header(text, relative_path)
            if header_text:
                header_chunk = Chunk(
                    text=header_text,
                    char_start=0,
                    char_end=0,
                    line_start=1,
                    line_end=1,
                    section_path="",
                    is_header=True,
                )
                chunks = [header_chunk] + chunks
        else:
            chunks = chunk_text(text, chunk_size, chunk_overlap, min_chunk_chars)

        if not chunks:
            stats.files_skipped += 1
            logger.info("Skip file %s: no chunks", path)
            continue

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = _embed_chunks(embedder, chunk_texts, batch_size)
        chunk_records: list[ChunkRecord] = []
        embedding_records: list[EmbeddingRecord] = []

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            text_hash = _hash_text(_normalize_text(chunk.text))
            token_count = _token_count(chunk.text)
            chunk_record = ChunkRecord(
                doc_id=doc_id,
                chunk_index=index,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                text=chunk.text,
                text_hash=text_hash,
                section_path=chunk.section_path,
                token_count=token_count,
                quality_flags={"header": chunk.is_header},
                is_header=chunk.is_header,
            )
            chunk_records.append(chunk_record)
            embedding_records.append(
                EmbeddingRecord(
                    chunk_id=chunk_record.chunk_id,
                    model_id=model_id,
                    dim=store.embedding_dim,
                    embedding=embedding,
                )
            )

        stats.chunks_created += len(chunk_records)
        inserted = store.insert_chunks_and_embeddings(chunk_records, embedding_records)
        stats.chunks_inserted += inserted
        stats.files_indexed += 1
        logger.debug("Indexed file %s: %s chunks", path, len(chunk_records))

    if incremental:
        existing_docs = store.list_documents(repo_id)
        to_delete = [
            doc["doc_id"]
            for doc in existing_docs
            if doc["path_norm"] not in seen_paths and doc.get("deleted_at") is None
        ]
        store.mark_documents_deleted(repo_id, to_delete)

    stats.duration_sec = time.monotonic() - start_time
    return stats


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _normalize_path(path: str) -> str:
    return path.replace("/", "\\").lower()


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _hash_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return digest


def _guess_language(suffix: str) -> Optional[str]:
    ext = suffix.lower()
    if ext in {".cs", ".csx"}:
        return "csharp"
    if ext in {".js"}:
        return "javascript"
    if ext in {".ts"}:
        return "typescript"
    if ext in {".py"}:
        return "python"
    if ext in {".md"}:
        return "markdown"
    if ext in {".html", ".htm"}:
        return "html"
    if ext in {".css"}:
        return "css"
    if ext in {".json", ".yaml", ".yml", ".toml", ".config"}:
        return "config"
    if ext in {".aspx", ".ashx", ".asp"}:
        return "aspnet"
    return None


def _guess_doc_type(source_type: str) -> str:
    if source_type == "md":
        return "markdown"
    if source_type == "code":
        return "code"
    if source_type == "config":
        return "config"
    return "text"


def _safe_stat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except OSError:
        return None


def _timestamp_or_none(stat: Optional[os.stat_result]) -> Optional[datetime]:
    if stat is None:
        return None
    return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _embed_chunks(
    embedder: Embedder, chunks: list[str], batch_size: int
) -> list[list[float]]:
    embed_many = getattr(embedder, "embed_many", None)
    if callable(embed_many):
        results: list[list[float]] = []
        safe_batch = max(1, batch_size)
        for start in range(0, len(chunks), safe_batch):
            batch = chunks[start : start + safe_batch]
            batch_embeddings = embed_many(batch)
            if len(batch_embeddings) != len(batch):
                raise RuntimeError("Embedding batch size mismatch")
            results.extend(batch_embeddings)
        return results
    return [embedder.embed(chunk) for chunk in chunks]


def _embedder_id(embedder: Embedder) -> str:
    model_name = getattr(embedder, "model_name", None)
    if model_name:
        return str(model_name)
    return embedder.__class__.__name__.lower()


def _get_env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


_CS_SYMBOL_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|sealed|abstract|static|partial)\s+)*"
    r"(class|interface|struct|enum)\s+(?P<name>[A-Za-z_]\w*)",
    re.MULTILINE,
)
_CS_NAMESPACE_RE = re.compile(r"^\s*namespace\s+(?P<name>[A-Za-z_][\w\.]*)", re.MULTILINE)
_CS_METHOD_RE = re.compile(
    r"^\s*(?:public|private|protected|internal)\s+"
    r"(?:static\s+)?(?:async\s+)?[\w<>\[\],\s]+\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)


def _build_code_header(text: str, relative_path: str) -> str:
    namespace = _first_match(_CS_NAMESPACE_RE, text)
    symbols = _extract_symbols(_CS_SYMBOL_RE, text)
    methods = _extract_methods(_CS_METHOD_RE, text, limit=40)
    lines: list[str] = [f"FILE: {relative_path}"]
    if namespace:
        lines.append(f"NAMESPACE: {namespace}")
    if symbols:
        classes = [name for kind, name in symbols if kind == "class"]
        interfaces = [name for kind, name in symbols if kind == "interface"]
        structs = [name for kind, name in symbols if kind == "struct"]
        enums = [name for kind, name in symbols if kind == "enum"]
        if classes:
            lines.append(f"CLASSES: {', '.join(classes[:30])}")
        if interfaces:
            lines.append(f"INTERFACES: {', '.join(interfaces[:30])}")
        if structs:
            lines.append(f"STRUCTS: {', '.join(structs[:30])}")
        if enums:
            lines.append(f"ENUMS: {', '.join(enums[:30])}")
    if methods:
        lines.append(f"METHODS: {', '.join(methods)}")
    tokens = _tokenize_path(relative_path)
    if tokens:
        lines.append(f"TOKENS: {' '.join(tokens)}")
    header = "\n".join(lines).strip()
    return header


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    return match.group("name").strip()


def _extract_symbols(pattern: re.Pattern[str], text: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        kind = match.group(1).strip().lower()
        name = match.group("name").strip()
        items.append((kind, name))
    return _dedupe(items)


def _extract_methods(
    pattern: re.Pattern[str], text: str, limit: int = 40
) -> list[str]:
    names: list[str] = []
    for match in pattern.finditer(text):
        name = match.group("name").strip()
        names.append(name)
        if len(names) >= limit:
            break
    return _dedupe(names)


def _tokenize_path(path: str) -> list[str]:
    raw = re.split(r"[\\/_.\-\s]+", path)
    tokens = [item for item in raw if item]
    return tokens[:30]


def _dedupe(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
