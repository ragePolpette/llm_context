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
from .store import ChunkRecord, DocumentRecord, EmbeddingRecord, RagStore, SymbolRecord


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
    symbol_search_enabled: bool = True,
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
        symbol_count = 0

        if symbol_search_enabled and source_type == "code":
            try:
                from .symbol_extractor import extract_symbols
                symbol_infos = extract_symbols(text, language or "")
                store.delete_symbols(doc_id)
                symbol_records = [
                    SymbolRecord(
                        doc_id=doc_id,
                        repo_id=repo_id,
                        name=si.name,
                        kind=si.kind,
                        namespace=si.namespace,
                        line_start=si.line_start,
                        line_end=si.line_end,
                        signature=si.signature,
                        language=language,
                    )
                    for si in symbol_infos
                ]
                if symbol_records:
                    store.insert_symbols(symbol_records)
                symbol_count = len(symbol_records)
            except Exception as _sym_exc:
                logger.warning("Symbol extraction failed for %s: %s", path, _sym_exc)

        if source_type == "md":
            chunks = chunk_markdown(text, md_chunk_size, chunk_overlap, min_chunk_chars)
        elif source_type == "code":
            chunks = chunk_code(text, code_chunk_size, min_chunk_chars)
            header_text = _build_code_header(text, relative_path, language=language)
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
        logger.info(
            "Ingest write path=%s repo=%s source_type=%s chunks_created=%s chunks_inserted=%s symbols=%s",
            relative_path,
            repo_id,
            source_type,
            len(chunk_records),
            inserted,
            symbol_count,
        )
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
_TS_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(class|interface|enum|type)\s+(?P<name>[A-Za-z_]\w*)",
    re.MULTILINE,
)
_PY_CLASS_RE = re.compile(
    r"^\s*class\s+(?P<name>[A-Za-z_]\w*)\s*(?:\(|:)",
    re.MULTILINE,
)
_PY_FUNC_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
_JS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_JS_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(",
    re.MULTILINE,
)
_JS_ASSIGNED_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
    re.MULTILINE,
)


def _build_code_header(
    text: str,
    relative_path: str,
    language: Optional[str] = None,
) -> str:
    language_norm = (language or "").strip().lower()
    namespace = (
        _first_match(_CS_NAMESPACE_RE, text)
        if language_norm in {"csharp", "aspnet", ""}
        else ""
    )
    symbols = _collect_code_symbols(text, language_norm)
    path_segments = _path_segments(relative_path)
    lines: list[str] = [f"FILE: {relative_path}"]
    if language_norm:
        lines.append(f"LANGUAGE: {language_norm}")
    if path_segments:
        lines.append(f"PROJECT: {path_segments[0]}")
    if len(path_segments) > 1:
        lines.append(f"SUBPROJECT: {path_segments[1]}")
    if namespace:
        lines.append(f"NAMESPACE: {namespace}")
    if symbols["classes"]:
        lines.append(f"CLASSES: {', '.join(symbols['classes'][:30])}")
    if symbols["interfaces"]:
        lines.append(f"INTERFACES: {', '.join(symbols['interfaces'][:30])}")
    if symbols["structs"]:
        lines.append(f"STRUCTS: {', '.join(symbols['structs'][:30])}")
    if symbols["enums"]:
        lines.append(f"ENUMS: {', '.join(symbols['enums'][:30])}")
    if symbols["types"]:
        lines.append(f"TYPES: {', '.join(symbols['types'][:30])}")
    if symbols["methods"]:
        lines.append(f"METHODS: {', '.join(symbols['methods'][:40])}")
    if symbols["functions"]:
        lines.append(f"FUNCTIONS: {', '.join(symbols['functions'][:40])}")
    tokens = _tokenize_path(relative_path)
    if tokens:
        lines.append(f"TOKENS: {' '.join(tokens)}")
    header = "\n".join(lines).strip()
    return header


def _collect_code_symbols(text: str, language: str) -> dict[str, list[str]]:
    classes: list[str] = []
    interfaces: list[str] = []
    structs: list[str] = []
    enums: list[str] = []
    types: list[str] = []
    methods: list[str] = []
    functions: list[str] = []

    if language in {"", "csharp", "aspnet"}:
        cs_symbols = _extract_symbols(_CS_SYMBOL_RE, text)
        classes.extend(name for kind, name in cs_symbols if kind == "class")
        interfaces.extend(name for kind, name in cs_symbols if kind == "interface")
        structs.extend(name for kind, name in cs_symbols if kind == "struct")
        enums.extend(name for kind, name in cs_symbols if kind == "enum")
        methods.extend(_extract_methods(_CS_METHOD_RE, text, limit=80))

    if language in {"", "typescript"}:
        ts_symbols = _extract_symbols(_TS_SYMBOL_RE, text)
        classes.extend(name for kind, name in ts_symbols if kind == "class")
        interfaces.extend(name for kind, name in ts_symbols if kind == "interface")
        enums.extend(name for kind, name in ts_symbols if kind == "enum")
        types.extend(name for kind, name in ts_symbols if kind == "type")

    if language in {"", "python"}:
        classes.extend(_extract_methods(_PY_CLASS_RE, text, limit=60))
        functions.extend(_extract_methods(_PY_FUNC_RE, text, limit=120))

    if language in {"", "javascript", "typescript"}:
        classes.extend(_extract_methods(_JS_CLASS_RE, text, limit=60))
        functions.extend(_extract_methods(_JS_FUNCTION_RE, text, limit=120))
        functions.extend(_extract_methods(_JS_ASSIGNED_FUNCTION_RE, text, limit=120))

    return {
        "classes": _dedupe(classes),
        "interfaces": _dedupe(interfaces),
        "structs": _dedupe(structs),
        "enums": _dedupe(enums),
        "types": _dedupe(types),
        "methods": _dedupe(methods),
        "functions": _dedupe(functions),
    }


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


def _path_segments(path: str) -> list[str]:
    segments = [item for item in re.split(r"[\\/]+", path.strip()) if item]
    if segments and "." in segments[-1]:
        segments = segments[:-1]
    return segments[:4]


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
