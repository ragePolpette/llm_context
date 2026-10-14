from rag_indexer import store as store_module
from rag_indexer.store import RagStore


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def fetchall(self):
        return []


class _FakeConn:
    def cursor(self):
        return _FakeCursor()


def test_query_vector_escapes_path_prefix(monkeypatch):
    captured = {}

    def fake_execute_params(cursor, sql_text, params):
        captured["sql_text"] = sql_text
        captured["params"] = params

    monkeypatch.setattr(store_module, "execute_params", fake_execute_params)
    store = RagStore(_FakeConn(), 4)

    results = store.query_vector(
        embedding=[0.1, 0.2, 0.3, 0.4],
        repo_id="repo",
        model_id="model",
        top_k=5,
        path_prefix=r"src\100%_match",
    )

    assert results == []
    assert "ESCAPE '\\'" in captured["sql_text"]
    assert r"src\\100\%\_match%" in captured["params"]


def test_query_keyword_escapes_path_prefix(monkeypatch):
    captured = {}

    def fake_execute_params(cursor, sql_text, params):
        captured["sql_text"] = sql_text
        captured["params"] = params

    monkeypatch.setattr(store_module, "execute_params", fake_execute_params)
    store = RagStore(_FakeConn(), 4)

    results = store.query_keyword(
        query_text="menu",
        repo_id="repo",
        top_k=5,
        path_prefix=r"src\100%_match",
    )

    assert results == []
    assert "ESCAPE '\\'" in captured["sql_text"]
    assert r"src\\100\%\_match%" in captured["params"]


def test_query_symbols_escapes_prefix_search(monkeypatch):
    captured = {}

    def fake_execute_params(cursor, sql_text, params):
        captured["sql_text"] = sql_text
        captured["params"] = params

    monkeypatch.setattr(store_module, "execute_params", fake_execute_params)
    store = RagStore(_FakeConn(), 4)

    results = store.query_symbols(name=r"Menu%_Service", exact=False)

    assert results == []
    assert "ESCAPE '\\'" in captured["sql_text"]
    assert r"Menu\%\_Service%" in captured["params"]





def test_path_prefix_filter_matches_legacy_backslash_rows(monkeypatch):
    captured = {}

    def fake_execute_params(cursor, sql_text, params):
        captured["sql_text"] = sql_text

    monkeypatch.setattr(store_module, "execute_params", fake_execute_params)
    store = RagStore(_FakeConn(), 4)
    store.query_keyword(query_text="menu", repo_id="repo", top_k=5, path_prefix="src/api/")

    assert "replace(d.path_norm, '\\', '/') LIKE" in captured["sql_text"]


def test_keyword_search_uses_configured_fts_config(monkeypatch):
    captured = {}

    def fake_execute_params(cursor, sql_text, params):
        captured["sql_text"] = sql_text
        captured["params"] = params

    monkeypatch.setattr(store_module, "execute_params", fake_execute_params)
    store = RagStore(_FakeConn(), 4, fts_config="english")
    store.query_keyword(query_text="menu", repo_id="repo", top_k=5)

    assert "italian" not in captured["sql_text"]
    assert "::regconfig" in captured["sql_text"]
    assert captured["params"][0] == "english"


def test_default_fts_config_is_simple():
    assert RagStore(_FakeConn(), 4).fts_config == "simple"


def test_chunk_insert_uses_configured_fts_config(monkeypatch):
    captured = {}

    def fake_execute_many_params(cursor, sql_text, rows):
        captured.setdefault("calls", []).append((sql_text, list(rows)))

    monkeypatch.setattr(store_module, "execute_many_params", fake_execute_many_params)

    class _CommitConn(_FakeConn):
        def commit(self):
            captured["committed"] = True

    store = RagStore(_CommitConn(), 2, fts_config="english")
    chunk = store_module.ChunkRecord(
        doc_id=store_module.uuid.uuid4(),
        chunk_index=0,
        char_start=0,
        char_end=4,
        line_start=1,
        line_end=1,
        text="text",
        text_hash="h",
    )
    embedding = store_module.EmbeddingRecord(chunk_id=chunk.chunk_id, model_id="m", dim=2, embedding=[0.1, 0.2])
    store.insert_chunks_and_embeddings([chunk], [embedding])

    chunk_sql, chunk_rows = captured["calls"][0]
    assert "to_tsvector(@Valore13::regconfig, @Valore14)" in chunk_sql
    assert chunk_rows[0][13] == "english"
    assert "italian" not in chunk_sql
