import pytest
import yaml

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


def test_symbol_search_falls_back_to_direct_connection_when_pool_connection_fails(
    monkeypatch, no_warmup
):
    captured = {}

    class FakeConn:
        def close(self):
            captured["closed"] = True

    class BrokenPoolContext:
        def __enter__(self):
            raise RuntimeError("pool broken")

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakePool:
        def connection(self):
            captured["used_pool"] = True
            return BrokenPoolContext()

    class FakeStore:
        def __init__(self, conn, embedding_dim):
            captured.setdefault("stores", 0)
            captured["stores"] += 1

        def query_symbols(self, **kwargs):
            captured["query"] = kwargs
            return [{"name": kwargs["name"]}]

    def fake_get_connection(dsn):
        captured["dsn"] = dsn
        return FakeConn()

    monkeypatch.setattr(mcp_handler_module, "get_pool", lambda dsn: FakePool())
    monkeypatch.setattr(mcp_handler_module, "get_connection", fake_get_connection)
    monkeypatch.setattr(mcp_handler_module, "RagStore", FakeStore)

    handler = MCPHandler()
    payload = handler._run_symbol_search({"name": "MenuService"})

    assert captured["used_pool"] is True
    assert captured["dsn"] == "postgresql://ctx_user:ctx_pass@localhost:5432/postgres"
    assert captured["closed"] is True
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


def _write_multi_project_config(tmp_path, *, multi_project_enabled=True, default_project_id="alpha"):
    projects_path = tmp_path / "projects.yaml"
    projects_path.write_text(
        yaml.safe_dump(
            {
                "projects": [
                    {
                        "project_id": "alpha",
                        "display_name": "Alpha",
                        "root_path": "repos/alpha",
                        "ingest_enabled": True,
                    },
                    {
                        "project_id": "beta",
                        "display_name": "Beta",
                        "root_path": "repos/beta",
                        "ingest_enabled": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "multi_project_enabled": multi_project_enabled,
                "default_project_id": default_project_id,
                "projects_registry_path": str(projects_path.name),
                "projects_state_path": "projects.state.json",
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_list_projects_returns_registered_projects(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))

    payload = handler._run_list_projects()

    assert payload["count"] == 2
    assert payload["multi_project_enabled"] is True
    assert [project["project_id"] for project in payload["projects"]] == ["alpha", "beta"]


def test_get_project_info_returns_registry_entry(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))

    payload = handler._run_get_project_info({"project_id": "alpha"})

    assert payload["project_id"] == "alpha"
    assert payload["display_name"] == "Alpha"
    assert payload["ingest_enabled"] is True


def test_multi_project_mode_requires_explicit_project_id(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())

    with pytest.raises(ValueError, match="requires explicit project_id"):
        handler._run_rag_context({"query_text": "bpofh"})


def test_single_project_mode_keeps_safe_default_fallback(monkeypatch, no_warmup, tmp_path):
    captured = {}

    def fake_build_context(**kwargs):
        captured["project_id"] = kwargs["project_id"]
        return "ctx", []

    monkeypatch.setattr(mcp_handler_module, "build_context", fake_build_context)
    handler = MCPHandler(
        config_path=str(
            _write_multi_project_config(
                tmp_path,
                multi_project_enabled=False,
                default_project_id="alpha",
            )
        )
    )
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())

    payload = handler._run_rag_context({"query_text": "bpofh"})

    assert captured["project_id"] == "alpha"
    assert payload["meta"]["project_id"] == "alpha"


def test_context_operational_status_exposes_project_summary(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))

    payload = handler.get_operational_status(ready=True)

    assert payload["status"] == "ready"
    assert payload["multi_project_enabled"] is True
    assert payload["project_count"] == 2
    assert payload["projects"][0]["project_id"] == "alpha"
