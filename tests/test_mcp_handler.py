import pytest

from rag_indexer import mcp_handler as mcp_handler_module
from rag_indexer.mcp_handler import MCPHandler, tool_rag_context, tool_symbol_search


@pytest.fixture(autouse=True)
def default_env(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_DSN", "postgresql://ctx_user:ctx_pass@localhost:5432/postgres")
    monkeypatch.setenv("LLM_CONTEXT_EMBEDDER", "local-hash")
    monkeypatch.setenv("LLM_CONTEXT_MAX_QUERY_EMBEDDING_ITEMS", "16")


@pytest.fixture
def no_warmup(monkeypatch):
    monkeypatch.setattr(MCPHandler, "_warmup_embedder", lambda self: None)


def test_handler_requires_default_dsn(monkeypatch, no_warmup):
    monkeypatch.delenv("LLM_CONTEXT_DSN", raising=False)

    with pytest.raises(RuntimeError, match="LLM_CONTEXT_DSN is required"):
        MCPHandler()


def test_tool_schemas_do_not_expose_dsn():
    rag_props = tool_rag_context()["inputSchema"]["properties"]
    symbol_props = tool_symbol_search()["inputSchema"]["properties"]

    assert "dsn" not in rag_props
    assert "dsn" not in symbol_props


def test_rag_context_uses_configured_default_dsn(monkeypatch, no_warmup):
    captured = {}

    def fake_build_context(**kwargs):
        captured["dsn"] = kwargs["dsn"]
        return "ctx", []

    monkeypatch.setattr(mcp_handler_module, "build_context", fake_build_context)
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())

    payload = handler._run_rag_context(
        {
            "query_text": "bpofh",
            "dsn": "postgresql://evil:evil@remote:5432/other",
        }
    )

    assert captured["dsn"] == "postgresql://ctx_user:ctx_pass@localhost:5432/postgres"
    assert payload["meta"]["project_id"] == "myproj"


def test_symbol_search_uses_configured_default_dsn(monkeypatch, no_warmup):
    captured = {}

    class FakeConn:
        def close(self):
            captured["closed"] = True

    class FakeStore:
        def __init__(self, conn, embedding_dim):
            captured["embedding_dim"] = embedding_dim

        def query_symbols(self, **kwargs):
            captured["query"] = kwargs
            return [{"name": kwargs["name"]}]

    def fake_get_connection(dsn):
        captured["dsn"] = dsn
        return FakeConn()

    monkeypatch.setattr(mcp_handler_module, "get_connection", fake_get_connection)
    monkeypatch.setattr(mcp_handler_module, "RagStore", FakeStore)

    handler = MCPHandler()
    payload = handler._run_symbol_search(
        {
            "name": "MenuService",
            "dsn": "postgresql://evil:evil@remote:5432/other",
        }
    )

    assert captured["dsn"] == "postgresql://ctx_user:ctx_pass@localhost:5432/postgres"
    assert captured["closed"] is True
    assert payload["count"] == 1


def test_symbol_search_uses_pool_when_available(monkeypatch, no_warmup):
    captured = {}

    class FakeConn:
        pass

    class FakePoolContext:
        def __enter__(self):
            captured["entered"] = True
            return FakeConn()

        def __exit__(self, exc_type, exc, tb):
            captured["exited"] = True
            return False

    class FakePool:
        def connection(self):
            captured["used_pool"] = True
            return FakePoolContext()

    class FakeStore:
        def __init__(self, conn, embedding_dim):
            captured["embedding_dim"] = embedding_dim

        def query_symbols(self, **kwargs):
            captured["query"] = kwargs
            return [{"name": kwargs["name"]}]

    def unexpected_get_connection(dsn):
        raise AssertionError("fallback connection path should not be used when pool exists")

    monkeypatch.setattr(mcp_handler_module, "get_pool", lambda dsn: FakePool())
    monkeypatch.setattr(mcp_handler_module, "get_connection", unexpected_get_connection)
    monkeypatch.setattr(mcp_handler_module, "RagStore", FakeStore)

    handler = MCPHandler()
    payload = handler._run_symbol_search({"name": "MenuService"})

    assert captured["used_pool"] is True
    assert captured["entered"] is True
    assert captured["exited"] is True
    assert payload["count"] == 1


def test_rag_context_rejects_oversized_query_embedding(monkeypatch, no_warmup):
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())

    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        handler._run_rag_context(
            {
                "query_embedding": [0.1] * 17,
                "embedding_dim": 17,
            }
        )


def test_rag_context_rejects_embedding_length_mismatch(monkeypatch, no_warmup):
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())

    with pytest.raises(ValueError, match="length mismatch"):
        handler._run_rag_context(
            {
                "query_embedding": [0.1] * 8,
                "embedding_dim": 4,
            }
        )
