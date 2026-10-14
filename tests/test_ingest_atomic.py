from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from rag_indexer import store as store_module
from rag_indexer.embedder import LocalHashEmbedder
from rag_indexer.ingest import ingest_project_v2
from rag_indexer.store import RagStore


class FakeStore:
    """In-memory stand-in for RagStore that models committed vs pending state."""

    embedding_dim = 8

    def __init__(self, documents: dict[str, dict] | None = None):
        self.committed = {key: dict(value) for key, value in (documents or {}).items()}
        self.pending: dict[str, dict] | None = None
        self.events: list[tuple[str, str, bool]] = []
        self.snapshots: list[dict[str, dict]] = []
        self.delete_repo_calls = 0

    @property
    def _state(self) -> dict[str, dict]:
        return self.pending if self.pending is not None else self.committed

    def _commit(self):
        if self.pending is None:
            self.snapshots.append({key: dict(value) for key, value in self.committed.items()})

    @contextmanager
    def atomic(self):
        self.pending = {key: dict(value) for key, value in self.committed.items()}
        try:
            yield
        except BaseException:
            self.pending = None
            raise
        self.committed = self.pending
        self.pending = None
        self.snapshots.append({key: dict(value) for key, value in self.committed.items()})

    def _by_doc_id(self, doc_id):
        for path_norm, doc in self._state.items():
            if doc["doc_id"] == doc_id:
                return path_norm, doc
        raise KeyError(doc_id)

    def has_symbol_refs_table(self):
        return False

    def delete_repo(self, repo_id):
        self.delete_repo_calls += 1
        self.committed.clear()
        return 0

    def get_document(self, repo_id, path_norm):
        doc = self.committed.get(path_norm)
        if doc is None:
            return None
        return {"doc_id": doc["doc_id"], "content_hash": doc["content_hash"], "deleted_at": None}

    def upsert_document(self, record):
        self.events.append(("upsert", record.path_norm, self.pending is not None))
        existing = self._state.get(record.path_norm)
        doc_id = existing["doc_id"] if existing else record.doc_id
        chunks = existing["chunks"] if existing else 0
        self._state[record.path_norm] = {"doc_id": doc_id, "content_hash": record.content_hash, "chunks": chunks}
        self._commit()
        return doc_id

    def delete_chunks(self, doc_id):
        path_norm, doc = self._by_doc_id(doc_id)
        self.events.append(("delete_chunks", path_norm, self.pending is not None))
        doc["chunks"] = 0
        self._commit()
        return 0

    def delete_symbols(self, doc_id):
        return 0

    def insert_symbols(self, records):
        return len(records)

    def insert_chunks_and_embeddings(self, chunks, embeddings):
        if not chunks:
            return 0
        path_norm, doc = self._by_doc_id(chunks[0].doc_id)
        self.events.append(("insert_chunks", path_norm, self.pending is not None))
        doc["chunks"] = len(chunks)
        self._commit()
        return len(chunks)

    def list_documents(self, repo_id):
        return [
            {"doc_id": doc["doc_id"], "path_norm": path_norm, "content_hash": doc["content_hash"], "deleted_at": None}
            for path_norm, doc in self.committed.items()
        ]

    def mark_documents_deleted(self, repo_id, doc_ids):
        return 0

    def delete_documents(self, repo_id, doc_ids):
        removed = [key for key, doc in self.committed.items() if doc["doc_id"] in set(doc_ids)]
        self.events.append(("delete_documents", ",".join(sorted(removed)), self.pending is not None))
        for key in removed:
            del self.committed[key]
        self._commit()
        return len(removed)


def _ingest(store, root: Path, *, incremental: bool):
    return ingest_project_v2(
        store=store,
        embedder=LocalHashEmbedder(8),
        repo_id="alpha",
        root=root,
        include_patterns=[".md", ".txt"],
        exclude_patterns=[],
        include_dirs=[],
        exclude_globs=[],
        never_index_ext=[],
        assets_template_only=False,
        assets_root_dir="",
        assets_template_dir="",
        max_bytes=100_000,
        chunk_size=400,
        chunk_overlap=0,
        md_chunk_size=400,
        code_chunk_size=400,
        min_chunk_chars=1,
        incremental=incremental,
        logger=logging.getLogger("test"),
        symbol_search_enabled=False,
    )


def _write_repo(root: Path) -> None:
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "a.md").write_text("# A\n\nalpha content " * 5, encoding="utf-8")
    (root / "docs" / "b.txt").write_text("beta content " * 5, encoding="utf-8")


def _existing_docs() -> dict[str, dict]:
    return {
        "docs/a.md": {"doc_id": uuid.uuid4(), "content_hash": "old", "chunks": 3},
        "docs/b.txt": {"doc_id": uuid.uuid4(), "content_hash": "old", "chunks": 2},
        "docs/removed.md": {"doc_id": uuid.uuid4(), "content_hash": "old", "chunks": 1},
    }


def test_full_reindex_never_exposes_an_empty_index(tmp_path):
    _write_repo(tmp_path)
    store = FakeStore(_existing_docs())

    stats = _ingest(store, tmp_path, incremental=False)

    assert store.delete_repo_calls == 0
    assert stats.files_indexed == 2
    assert stats.deleted_rows == 1
    assert set(store.committed) == {"docs/a.md", "docs/b.txt"}
    for snapshot in store.snapshots:
        assert snapshot, "readers observed an empty index"
        for path_norm in ("docs/a.md", "docs/b.txt"):
            assert snapshot[path_norm]["chunks"] > 0, f"readers observed {path_norm} without chunks"


def test_full_reindex_writes_each_document_in_one_transaction(tmp_path):
    _write_repo(tmp_path)
    store = FakeStore(_existing_docs())

    _ingest(store, tmp_path, incremental=False)

    document_writes = [event for event in store.events if event[0] in {"upsert", "delete_chunks", "insert_chunks"}]
    assert document_writes
    assert all(inside_transaction for _, _, inside_transaction in document_writes)


def test_full_reindex_removes_stale_documents_only_at_the_end(tmp_path):
    _write_repo(tmp_path)
    store = FakeStore(_existing_docs())

    _ingest(store, tmp_path, incremental=False)

    assert store.events[-1][0] == "delete_documents"
    assert store.events[-1][1] == "docs/removed.md"
    assert "docs/removed.md" in store.snapshots[-2]


def test_full_reindex_reprocesses_unchanged_files(tmp_path):
    _write_repo(tmp_path)
    store = FakeStore()
    _ingest(store, tmp_path, incremental=False)
    store.events.clear()

    stats = _ingest(store, tmp_path, incremental=False)

    assert stats.files_indexed == 2
    assert stats.deleted_rows == 0


def test_incremental_ingest_skips_unchanged_files(tmp_path):
    _write_repo(tmp_path)
    store = FakeStore()
    _ingest(store, tmp_path, incremental=True)
    store.events.clear()

    stats = _ingest(store, tmp_path, incremental=True)

    assert stats.files_indexed == 0
    assert stats.files_skipped == 2
    assert store.events == []


class _FakeCursor:
    rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _TxConn:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _FakeCursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_rag_store_atomic_commits_once(monkeypatch):
    monkeypatch.setattr(store_module, "execute_params", lambda cursor, sql_text, params: None)
    conn = _TxConn()
    store = RagStore(conn, 4)

    with store.atomic():
        store.delete_chunks(uuid.uuid4())
        store.delete_symbols(uuid.uuid4())
        assert conn.commits == 0

    assert conn.commits == 1
    store.delete_chunks(uuid.uuid4())
    assert conn.commits == 2


def test_rag_store_atomic_rolls_back_on_error(monkeypatch):
    monkeypatch.setattr(store_module, "execute_params", lambda cursor, sql_text, params: None)
    conn = _TxConn()
    store = RagStore(conn, 4)

    with pytest.raises(RuntimeError):
        with store.atomic():
            store.delete_chunks(uuid.uuid4())
            raise RuntimeError("boom")

    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_rag_store_delete_documents_targets_given_ids(monkeypatch):
    captured = {}

    def fake_execute_params(cursor, sql_text, params):
        captured["sql_text"] = sql_text
        captured["params"] = params

    monkeypatch.setattr(store_module, "execute_params", fake_execute_params)
    conn = _TxConn()
    doc_id = uuid.uuid4()

    assert RagStore(conn, 4).delete_documents("alpha", []) == 0
    assert RagStore(conn, 4).delete_documents("alpha", [doc_id]) == 1
    assert captured["sql_text"].startswith("DELETE FROM documents WHERE repo_id")
    assert captured["params"] == ["alpha", [doc_id]]
