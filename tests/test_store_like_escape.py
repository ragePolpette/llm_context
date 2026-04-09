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



