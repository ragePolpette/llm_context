import pytest
import yaml

from rag_indexer import mcp_handler as mcp_handler_module
from rag_indexer.mcp_handler import (
    MCPHandler,
    format_functional_context_text,
    format_tool_text,
    tool_map_work_item_to_codebase,
    tool_rag_context,
    tool_symbol_search,
)


@pytest.fixture(autouse=True)
def default_env(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_DSN", "postgresql://ctx_user:ctx_pass@localhost:5432/postgres")
    monkeypatch.setenv("LLM_CONTEXT_EMBEDDER", "local-hash")
    monkeypatch.setenv("LLM_CONTEXT_MAX_QUERY_EMBEDDING_ITEMS", "16")
    monkeypatch.setattr(
        mcp_handler_module,
        "inspect_database_runtime",
        lambda dsn, connect_timeout=3: {
            "reachable": True,
            "database": "postgres",
            "server_version": "16.0",
            "pgvector_available": True,
            "schema_ready": True,
            "required_tables": {
                "documents": True,
                "chunks": True,
                "chunk_embeddings": True,
                "index_runs": True,
                "symbols": True,
            },
            "error": None,
        },
    )


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
    mapping_props = tool_map_work_item_to_codebase()["inputSchema"]["properties"]

    assert "dsn" not in rag_props
    assert "dsn" not in symbol_props
    assert "dsn" not in mapping_props


def test_rag_context_uses_configured_default_dsn(monkeypatch, no_warmup):
    captured = {}

    def fake_build_context(**kwargs):
        captured["dsn"] = kwargs["dsn"]
        return "ctx", []

    monkeypatch.setattr(mcp_handler_module, "build_context", fake_build_context)
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())
    monkeypatch.setattr(handler, "_collect_context_symbols", lambda **kwargs: [])

    payload = handler._run_rag_context(
        {
            "query_text": "legacylib",
            "dsn": "postgresql://evil:evil@remote:5432/other",
        }
    )

    assert captured["dsn"] == "postgresql://ctx_user:ctx_pass@localhost:5432/postgres"
    assert payload["meta"]["project_id"] == "myproj"
    assert "functional_context" in payload


def test_rag_context_retries_without_auto_scope_when_derived_prefix_has_no_hits(monkeypatch, no_warmup):
    calls = []

    def fake_build_context(**kwargs):
        calls.append(kwargs["path_prefix"])
        if len(calls) == 1:
            return "ctx-empty", []
        return "ctx-final", [{"path": "src/api/controllers/BillingController.cs", "score": 0.91, "content": "match"}]

    monkeypatch.setattr(mcp_handler_module, "build_context", fake_build_context)
    monkeypatch.setattr(mcp_handler_module, "resolve_path_prefix", lambda **kwargs: "src/api/controllers/BillingController.cs")
    monkeypatch.setattr(mcp_handler_module, "format_context_sheet", lambda **kwargs: "sheet")
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())
    monkeypatch.setattr(handler, "_collect_context_symbols", lambda **kwargs: [])

    payload = handler._run_rag_context({"query_text": "GenerateInvoice fatturazione studi"})

    assert calls == ["src/api/controllers/BillingController.cs", None]
    assert payload["meta"]["auto_scope_fallback_used"] is True
    assert payload["meta"]["initial_path_prefix"] == "src/api/controllers/BillingController.cs"
    assert payload["meta"]["path_prefix"] is None


def test_rag_context_does_not_retry_when_path_prefix_is_explicit(monkeypatch, no_warmup):
    calls = []

    def fake_build_context(**kwargs):
        calls.append(kwargs["path_prefix"])
        return "ctx-empty", []

    monkeypatch.setattr(mcp_handler_module, "build_context", fake_build_context)
    monkeypatch.setattr(mcp_handler_module, "format_context_sheet", lambda **kwargs: "sheet")
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())
    monkeypatch.setattr(handler, "_collect_context_symbols", lambda **kwargs: [])

    payload = handler._run_rag_context(
        {"query_text": "GenerateInvoice", "path_prefix": "src/api/controllers/BillingController.cs"}
    )

    assert calls == ["src\\api\\controllers\\BillingController.cs"]
    assert payload["meta"]["auto_scope_fallback_used"] is False


def test_rag_context_builds_symbol_follow_up_from_core_file_roles(monkeypatch, no_warmup):
    retrieval_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.92,
            "text_hash": "controller-hit",
            "line_start": 10,
            "line_end": 30,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Billing API entry point.",
            "text": "public class BillingController { ... }",
        },
        {
            "source_path": "src/domain/services/BillingService.cs",
            "score": 0.88,
            "text_hash": "service-hit",
            "line_start": 50,
            "line_end": 90,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Service that executes GenerateInvoice.",
            "text": "public class BillingService { ... }",
        },
        {
            "source_path": "src/api/contracts/IBillingService.cs",
            "score": 0.81,
            "text_hash": "contract-hit",
            "line_start": 1,
            "line_end": 20,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Contract for the billing API.",
            "text": "public interface IBillingService { ... }",
        },
    ]
    symbol_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "name": "GenerateInvoice",
            "kind": "method",
            "signature": "public void GenerateInvoice()",
            "line_start": 44,
            "line_end": 60,
        },
        {
            "source_path": "src/domain/services/BillingService.cs",
            "name": "BillingService",
            "kind": "class",
            "signature": "public class BillingService",
            "line_start": 1,
            "line_end": 120,
        },
        {
            "source_path": "src/api/contracts/IBillingService.cs",
            "name": "IBillingService",
            "kind": "interface",
            "signature": "public interface IBillingService",
            "line_start": 1,
            "line_end": 40,
        },
    ]

    monkeypatch.setattr(mcp_handler_module, "build_context", lambda **kwargs: ("ctx", retrieval_results))
    monkeypatch.setattr(mcp_handler_module, "format_context_sheet", lambda **kwargs: "sheet")
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())
    monkeypatch.setattr(handler, "_collect_context_symbols", lambda **kwargs: symbol_results)

    payload = handler._run_rag_context({"query_text": "GenerateInvoice billing api"})

    symbol_follow_up = payload["functional_context"]["tool_hints"]["symbol_follow_up"]
    suggested_queries = symbol_follow_up["suggested_queries"]

    assert [item["name"] for item in suggested_queries[:3]] == [
        "GenerateInvoice",
        "BillingService",
        "IBillingService",
    ]
    assert [item["source_role"] for item in suggested_queries[:3]] == [
        "entry_point",
        "implementation",
        "contract",
    ]
    assert all(item["exact"] is True for item in suggested_queries[:3])
    assert symbol_follow_up["focus_paths"] == [
        "src/api/controllers/BillingController.cs",
        "src/domain/services/BillingService.cs",
        "src/api/contracts/IBillingService.cs",
    ]
    assert payload["functional_context"]["tool_hints"]["recommended_follow_up"][0]["tool"] == "symbol_search"
    assert payload["functional_context"]["tool_hints"]["recommended_follow_up"][0]["suggested_queries"][0]["name"] == "GenerateInvoice"


def test_rag_context_symbol_follow_up_falls_back_to_query_candidates(monkeypatch, no_warmup):
    monkeypatch.setattr(mcp_handler_module, "build_context", lambda **kwargs: ("ctx", []))
    monkeypatch.setattr(mcp_handler_module, "format_context_sheet", lambda **kwargs: "sheet")
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())
    monkeypatch.setattr(handler, "_collect_context_symbols", lambda **kwargs: [])

    payload = handler._run_rag_context({"query_text": "Apri GenerateInvoice e IFatture"})

    suggested_queries = payload["functional_context"]["tool_hints"]["symbol_follow_up"]["suggested_queries"]

    assert [item["name"] for item in suggested_queries] == ["GenerateInvoice", "IFatture"]
    assert all(item["exact"] is False for item in suggested_queries)
    assert all(item["source_role"] == "query_candidate" for item in suggested_queries)


def test_rag_search_returns_investigation_payload(monkeypatch, no_warmup):
    retrieval_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.94,
            "text_hash": "a1",
            "line_start": 10,
            "line_end": 30,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Controller principale per GenerateInvoice.",
            "text": "public class FatturaController { ... }",
        },
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "score": 0.88,
            "text_hash": "a2",
            "line_start": 44,
            "line_end": 60,
            "chunk_index": 1,
            "section_path": "",
            "snippet": "Metodo GenerateInvoice.",
            "text": "public void GenerateInvoice() { ... }",
        },
        {
            "source_path": "src/domain/services/InvoiceService.cs",
            "score": 0.82,
            "text_hash": "b1",
            "line_start": 5,
            "line_end": 25,
            "chunk_index": 0,
            "section_path": "",
            "snippet": "Implementazione del servizio fatture.",
            "text": "public class InvoiceService { ... }",
        },
    ]
    symbol_results = [
        {
            "source_path": "src/api/controllers/BillingController.cs",
            "name": "GenerateInvoice",
            "kind": "method",
            "signature": "public void GenerateInvoice()",
            "line_start": 44,
            "line_end": 60,
        }
    ]

    monkeypatch.setattr(mcp_handler_module, "build_context", lambda **kwargs: ("ctx", retrieval_results))
    monkeypatch.setattr(mcp_handler_module, "format_context_sheet", lambda **kwargs: "sheet")
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())
    monkeypatch.setattr(handler, "_collect_context_symbols", lambda **kwargs: symbol_results)

    payload = handler._run_rag_search({"query_text": "GenerateInvoice fattura api"})

    assert payload["summary"]["result_count"] == 3
    assert payload["summary"]["unique_file_count"] == 2
    assert payload["summary"]["coverage"] == "focused_multi_file"
    assert payload["meta"]["search_mode"] == "investigation"
    assert payload["result_groups"][0]["source_path"] == "src/api/controllers/BillingController.cs"
    assert payload["result_groups"][0]["hit_count"] == 2
    assert payload["investigation_hints"]["suggested_files"][0] == "src/api/controllers/BillingController.cs"
    assert payload["investigation_hints"]["suggested_symbol_queries"][0]["name"] == "GenerateInvoice"


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


def test_context_info_exposes_tool_map_and_usage_notes(no_warmup):
    handler = MCPHandler()

    payload = handler._run_context_info()

    assert payload["runtime_name"] == "default"
    assert payload["config_path"].endswith("config.yaml")
    assert payload["storage_target"]["database"] == "postgres"
    assert payload["storage_target"]["dedicated_candidate"] is False
    assert payload["database_runtime"]["reachable"] is True
    assert payload["database_runtime"]["deployment_hint"] == "local_or_docker_port_mapping"
    assert "runtime_readiness" in payload
    assert payload["runtime_readiness"]["status"] in {"degraded", "blocked", "ready"}
    assert "recommended_actions" in payload["runtime_readiness"]
    assert "tool_map" in payload
    assert "tool_roles" in payload
    assert "quick_start" in payload
    assert "decision_guide" in payload
    assert "anti_patterns" in payload
    assert "recommended_workflows" in payload
    assert "rag_context" in payload["tool_map"]["working_context"]
    assert payload["tool_roles"]["symbol_search"]["role"] == "precision lookup"
    assert payload["quick_start"][0]["tool"] == "context_info"
    assert payload["decision_guide"][0]["use"] == ["rag_context"]
    assert payload["tool_roles"]["rag_search"]["returns"] == [
        "results",
        "summary",
        "result_groups",
        "investigation_hints",
        "meta",
    ]
    assert payload["usage_notes"]["rag_search"].startswith("Tool di approfondimento/raw")


def test_format_functional_context_text_includes_symbol_follow_up_section():
    text = format_functional_context_text(
        {
            "query": {"text": "GenerateInvoice", "path_prefix": None},
            "summary": {"core_file_count": 1, "supporting_match_count": 0, "symbol_hit_count": 2},
            "entry_points": [],
            "core_files": [],
            "supporting_matches": [],
            "assembled_context": "",
            "tool_hints": {
                "symbol_follow_up": {
                    "suggested_queries": [
                        {
                            "name": "GenerateInvoice",
                            "kind": "method",
                            "exact": True,
                            "source_path": "src/api/controllers/BillingController.cs",
                            "source_role": "entry_point",
                            "reason": "Conferma il simbolo di ingresso.",
                        }
                    ]
                },
                "recommended_follow_up": [],
            },
        }
    )

    assert "SYMBOL FOLLOW-UP" in text
    assert "method GenerateInvoice exact=true role=entry_point" in text
    assert "path=src/api/controllers/BillingController.cs" in text


def test_format_tool_text_formats_context_info_as_decision_guide():
    text = format_tool_text(
        "context_info",
        {},
        {
            "server": "llm-context-mcp",
            "runtime_name": "llm-context",
            "multi_project_enabled": True,
            "default_project_id": "llm_context",
            "database_runtime": {
                "reachable": True,
                "database": "llm_context",
                "deployment_hint": "remote_or_managed_postgres",
                "network_scope": "remote_hostname",
                "server_version": "16.0",
                "pgvector_available": True,
                "schema_ready": True,
            },
            "runtime_readiness": {
                "status": "blocked",
                "ready_for_queries": False,
                "summary": "Runtime non pronto: nessun progetto con integrity=ok.",
                "blocking_reasons": ["no_project_with_integrity_ok"],
                "warnings": ["registered_projects_are_not_indexed"],
                "recommended_actions": ["Eseguire ingest sul progetto registrato."],
            },
            "quick_start": [
                {"step": "1", "tool": "context_info", "reason": "scegli il flusso corretto"}
            ],
            "decision_guide": [
                {
                    "if": "devi il package principale",
                    "use": ["rag_context"],
                    "because": "restituisce il contesto operativo",
                }
            ],
            "anti_patterns": ["non usare rag_search come primo tool"],
            "recommended_workflows": [
                {"goal": "iniziare", "steps": ["context_info", "rag_context"]}
            ],
        },
    )

    assert "CONTEXT INFO" in text
    assert "RUNTIME READINESS" in text
    assert "DATABASE RUNTIME" in text
    assert "deployment_hint: remote_or_managed_postgres network_scope=remote_hostname" in text
    assert "status: blocked ready_for_queries=false" in text
    assert "blocker: no_project_with_integrity_ok" in text
    assert "QUICK START" in text
    assert "DECISION GUIDE" in text
    assert "ANTI-PATTERNS" in text
    assert "if devi il package principale -> use rag_context" in text


def test_format_tool_text_formats_rag_search_as_investigation_summary():
    text = format_tool_text(
        "rag_search",
        {},
        {
            "query": {"text": "GenerateInvoice", "path_prefix": None},
            "summary": {
                "result_count": 2,
                "unique_file_count": 1,
                "coverage": "single_hotspot",
                "top_score": 0.94,
            },
            "result_groups": [
                {
                    "source_path": "src/api/controllers/BillingController.cs",
                    "hit_count": 2,
                    "top_score": 0.94,
                    "line_spans": ["L10-L30", "L44-L60"],
                }
            ],
            "investigation_hints": {
                "suggested_files": ["src/api/controllers/BillingController.cs"],
                "suggested_symbol_queries": [
                    {"name": "GenerateInvoice", "kind": "method", "exact": True}
                ],
            },
            "results": [
                {
                    "source_path": "src/api/controllers/BillingController.cs",
                    "score": 0.94,
                    "line_start": 44,
                    "line_end": 60,
                    "snippet": "Metodo GenerateInvoice.",
                }
            ],
        },
    )

    assert "RAG SEARCH" in text
    assert "TOP FILES" in text
    assert "INVESTIGATION HINTS" in text
    assert "symbol_search: method GenerateInvoice exact=true" in text


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
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T11:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T11:00:00Z",
            "indexed_documents": 6,
            "index_fingerprint": "idx-alpha",
            "store_target": {"database": "postgres", "dsn_fingerprint": "9c6fe00d4d62"},
        },
    )

    payload = handler._run_list_projects()

    assert payload["count"] == 2
    assert payload["multi_project_enabled"] is True
    assert payload["projects"][0]["index_manifest"]["indexed_documents"] == 6
    assert payload["projects"][0]["integrity"]["status"] == "ok"


def test_map_work_item_to_codebase_returns_structured_mapping(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))

    monkeypatch.setattr(
        handler,
        "_run_rag_context",
        lambda args, tool_name="rag_context": {
            "functional_context": {
                "summary": {
                    "core_file_count": 2,
                    "supporting_match_count": 1,
                    "symbol_hit_count": 1,
                },
                "entry_points": [
                    {
                        "name": "GenerateInvoice",
                        "kind": "method",
                        "source_path": "pubblico\\api\\Controllers\\Fattura.cs",
                        "line_start": 120,
                        "line_end": 180,
                    }
                ],
                "core_files": [
                    {
                        "source_path": "pubblico\\api\\Controllers\\Fattura.cs",
                        "aggregate_score": 1.75,
                        "max_score": 0.9,
                        "match_count": 3,
                        "symbol_hits": [{"name": "GenerateInvoice"}],
                    },
                    {
                        "source_path": "pubblico\\api\\Services\\Fatture\\Generator.cs",
                        "aggregate_score": 0.65,
                        "max_score": 0.5,
                        "match_count": 1,
                        "symbol_hits": [],
                    },
                ],
                "supporting_matches": [],
            }
        },
    )

    payload = handler._run_map_work_item_to_codebase(
        {
            "ticket_key": "BPO-123",
            "summary": "Errore generazione fattura",
            "description": "La fatturazione studio fallisce in alcuni casi.",
            "product_target_hint": "legacy",
            "project_id": "alpha",
        }
    )

    assert payload["ticket_key"] == "BPO-123"
    assert payload["project_id"] == "alpha"
    assert payload["product_target"] == "legacy"
    assert payload["repo_target"] == "alpha"
    assert payload["in_scope"] is True
    assert payload["feasibility"] in {"high", "medium"}
    assert "GenerateInvoice" in payload["implementation_hint"]
    assert payload["paths"][0] == "pubblico\\api\\Controllers\\Fattura.cs"


def test_map_work_item_to_codebase_accepts_workspace_root(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))

    monkeypatch.setattr(
        handler,
        "_run_rag_context",
        lambda args, tool_name="rag_context": {
            "functional_context": {
                "summary": {
                    "core_file_count": 0,
                    "supporting_match_count": 0,
                    "symbol_hit_count": 0,
                },
                "entry_points": [],
                "core_files": [],
                "supporting_matches": [],
            }
        },
    )

    alpha_root = handler._project_registry.require_project("alpha").root_path
    payload = handler._run_map_work_item_to_codebase(
        {
            "summary": "Ticket senza hit",
            "workspace_root": str(alpha_root),
        }
    )

    assert payload["project_id"] == "alpha"
    assert payload["in_scope"] is False
    assert payload["feasibility"] in {"blocked", "out_of_scope"}
    assert payload["blockers"]


def test_get_project_info_returns_registry_entry(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))

    payload = handler._run_get_project_info({"project_id": "alpha"})

    assert payload["project_id"] == "alpha"
    assert payload["display_name"] == "Alpha"
    assert payload["ingest_enabled"] is True


def test_get_project_info_exposes_index_manifest(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T10:30:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "schema_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T10:30:00Z",
            "indexed_documents": 5,
            "indexed_chunks": 18,
            "indexed_symbols": 3,
            "index_fingerprint": "idx-alpha",
            "store_target": {"database": "postgres"},
        },
    )

    payload = handler._run_get_project_info({"project_id": "alpha"})

    assert payload["index_manifest"] is not None
    assert payload["index_manifest"]["present"] is True
    assert payload["index_manifest"]["indexed_chunks"] == 18
    assert payload["integrity"]["status"] == "ok"


def test_get_project_info_marks_missing_manifest_as_unreliable(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T12:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )

    payload = handler._run_get_project_info({"project_id": "alpha"})

    assert payload["index_manifest"] is None
    assert payload["integrity"]["status"] == "unreliable"
    assert "runtime_state_has_index_but_manifest_missing" in payload["integrity"]["reasons"]


def test_multi_project_mode_requires_explicit_project_id(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())

    with pytest.raises(ValueError, match="requires explicit project_id"):
        handler._run_rag_context({"query_text": "legacylib"})


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
    monkeypatch.setattr(handler, "_collect_context_symbols", lambda **kwargs: [])

    payload = handler._run_rag_context({"query_text": "legacylib"})

    assert captured["project_id"] == "alpha"
    assert payload["meta"]["project_id"] == "alpha"


def test_rag_context_default_format_returns_functional_text():
    text = format_tool_text(
        "rag_context",
        {},
        {
            "functional_context": {
                "query": {"text": "fatturazione studi", "path_prefix": "pubblico\\api"},
                "summary": {
                    "core_file_count": 1,
                    "supporting_match_count": 2,
                    "symbol_hit_count": 1,
                },
                "entry_points": [
                    {
                        "kind": "method",
                        "name": "GenerateInvoice",
                        "source_path": "src/api/controllers/BillingController.cs",
                        "line_start": 10,
                        "line_end": 30,
                    }
                ],
                "core_files": [
                    {
                        "source_path": "src/api/controllers/BillingController.cs",
                        "aggregate_score": 1.2,
                        "match_count": 2,
                        "symbol_hits": [{}],
                    }
                ],
                "supporting_matches": [],
                "tool_hints": {
                    "recommended_follow_up": [
                        {
                            "tool": "symbol_search",
                            "reason": "Usa symbol_search per confermare signature.",
                            "suggested_names": ["GenerateInvoice"],
                        }
                    ]
                },
                "assembled_context": "FILE src/api/controllers/BillingController.cs\n...",
            },
            "context": "legacy ctx",
            "context_sheet": "legacy sheet",
        },
    )

    assert text.startswith("FUNCTIONAL CONTEXT")
    assert "ENTRY POINTS" in text
    assert "FOLLOW-UP TOOLS" in text
    assert "ASSEMBLED CONTEXT" in text


def test_rag_context_payload_exposes_tool_hints(monkeypatch, no_warmup):
    def fake_build_context(**kwargs):
        return "ctx", [{"source_path": "a.cs", "score": 0.9, "text": "x"}]

    handler = MCPHandler()
    monkeypatch.setattr(mcp_handler_module, "build_context", fake_build_context)
    monkeypatch.setattr(mcp_handler_module, "format_context_sheet", lambda **kwargs: "sheet")
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())
    monkeypatch.setattr(
        handler,
        "_collect_context_symbols",
        lambda **kwargs: [
            {
                "name": "GenerateInvoice",
                "kind": "method",
                "source_path": "src/api/controllers/BillingController.cs",
                "line_start": 10,
                "line_end": 20,
                "signature": "void GenerateInvoice()",
            }
        ],
    )

    payload = handler._run_rag_context({"query_text": "GenerateInvoice"})

    tool_hints = payload["functional_context"]["tool_hints"]
    assert tool_hints["primary_tool"] == "rag_context"
    assert tool_hints["recommended_follow_up"][0]["tool"] == "symbol_search"
    assert "GenerateInvoice" in tool_hints["recommended_follow_up"][0]["suggested_names"]


def test_rag_context_legacy_format_returns_legacy_context():
    text = format_tool_text(
        "rag_context",
        {"format": "legacy"},
        {
            "functional_context": {"assembled_context": "new"},
            "context": "legacy ctx",
            "context_sheet": "legacy sheet",
        },
    )

    assert text == "legacy ctx"


def test_rag_search_returns_raw_results_without_formatted_context(monkeypatch, no_warmup):
    def fake_build_context(**kwargs):
        return "ctx", [{"source_path": "a.cs", "score": 0.9, "text": "x"}]

    monkeypatch.setattr(mcp_handler_module, "build_context", fake_build_context)
    handler = MCPHandler()
    monkeypatch.setattr(handler, "_get_embedder", lambda args, embedding_dim: object())
    monkeypatch.setattr(handler, "_collect_context_symbols", lambda **kwargs: [{"name": "X"}])

    payload = handler._run_rag_search({"query_text": "fatturazione"})

    assert "context" not in payload
    assert "context_sheet" not in payload
    assert "functional_context" not in payload
    assert "symbol_results" not in payload
    assert payload["results"][0]["source_path"] == "a.cs"


def test_context_operational_status_exposes_project_summary(monkeypatch, no_warmup, tmp_path):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T11:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T11:00:00Z",
            "indexed_documents": 7,
            "indexed_chunks": 30,
            "indexed_symbols": 4,
            "index_fingerprint": "idx-alpha",
        },
    )

    payload = handler.get_operational_status(ready=True)

    assert payload["status"] == "ready"
    assert payload["runtime_name"] == "default"
    assert payload["storage_target"]["database"] == "postgres"
    assert payload["project_manifest_dir"].endswith("project_manifests")
    assert payload["multi_project_enabled"] is True
    assert payload["project_count"] == 2
    assert payload["projects"][0]["project_id"] == "alpha"
    assert payload["projects"][0]["index_manifest"]["indexed_documents"] == 7
    assert payload["projects"][0]["integrity"]["status"] == "ok"


def test_context_operational_status_marks_storage_target_mismatch_as_stale(
    monkeypatch, no_warmup, tmp_path
):
    monkeypatch.setenv(
        "LLM_CONTEXT_DSN",
        "postgresql://ctx_user:ctx_pass@localhost:5432/llm_context",
    )
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T13:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T13:00:00Z",
            "index_fingerprint": "idx-alpha",
            "store_target": {"database": "postgres"},
        },
    )

    payload = handler.get_operational_status(ready=True)

    assert payload["projects"][0]["integrity"]["status"] == "stale"
    assert "runtime_storage_target_differs_from_manifest" in payload["projects"][0]["integrity"]["reasons"]
    assert payload["projects"][0]["integrity"]["store_target_match"] is False
    assert payload["runtime_readiness"]["status"] == "blocked"
    assert "registered_projects_have_stale_index_state" in payload["runtime_readiness"]["warnings"]


def test_context_operational_status_marks_runtime_readiness_ready(
    monkeypatch, no_warmup, tmp_path
):
    monkeypatch.setenv(
        "LLM_CONTEXT_DSN",
        "postgresql://ctx_user:ctx_pass@localhost:5432/llm_context",
    )
    monkeypatch.setenv("LLM_CONTEXT_STORE_TARGET", "llm-context-local-pg")
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T13:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T13:00:00Z",
            "index_fingerprint": "idx-alpha",
            "store_target": {
                "name": "llm-context-local-pg",
                "database": "llm_context",
            },
        },
    )
    handler._project_registry.save_runtime_state(
        "beta",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T13:00:00Z",
        index_version="v2",
        index_fingerprint="idx-beta",
    )
    handler._project_registry.save_index_manifest(
        "beta",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T13:00:00Z",
            "index_fingerprint": "idx-beta",
            "store_target": {
                "name": "llm-context-local-pg",
                "database": "llm_context",
            },
        },
    )

    payload = handler.get_operational_status(ready=True)

    assert payload["runtime_readiness"]["status"] == "ready"
    assert payload["runtime_readiness"]["ready_for_queries"] is True
    assert payload["runtime_readiness"]["queryable_projects"] == ["alpha", "beta"]
    assert payload["runtime_readiness"]["project_integrity_counts"]["ok"] == 2


def test_context_operational_status_marks_database_unreachable_as_blocked(
    monkeypatch, no_warmup, tmp_path
):
    monkeypatch.setenv(
        "LLM_CONTEXT_DSN",
        "postgresql://ctx_user:ctx_pass@db-host:5432/llm_context",
    )
    monkeypatch.setenv("LLM_CONTEXT_STORE_TARGET", "llm-context-local-pg")
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T13:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T13:00:00Z",
            "index_fingerprint": "idx-alpha",
            "store_target": {
                "name": "llm-context-local-pg",
                "database": "llm_context",
            },
        },
    )
    monkeypatch.setattr(
        handler,
        "_get_database_runtime_summary",
        lambda refresh=False: {
            "reachable": False,
            "database": None,
            "server_version": None,
            "pgvector_available": None,
            "schema_ready": None,
            "required_tables": {},
            "error": "OperationalError: connection refused",
        },
    )

    payload = handler.get_operational_status(ready=True)

    assert payload["database_runtime"]["reachable"] is False
    assert payload["database_runtime"]["deployment_hint"] == "remote_or_managed_postgres"
    assert payload["runtime_readiness"]["status"] == "blocked"
    assert "database_unreachable" in payload["runtime_readiness"]["blocking_reasons"]
    assert any(
        "Postgres remoto o cloud" in item
        for item in payload["runtime_readiness"]["recommended_actions"]
    )


def test_context_operational_status_marks_pgvector_missing_as_blocked(
    monkeypatch, no_warmup, tmp_path
):
    monkeypatch.setenv(
        "LLM_CONTEXT_DSN",
        "postgresql://ctx_user:ctx_pass@db-host:5432/llm_context",
    )
    monkeypatch.setenv("LLM_CONTEXT_STORE_TARGET", "llm-context-local-pg")
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T13:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T13:00:00Z",
            "index_fingerprint": "idx-alpha",
            "store_target": {
                "name": "llm-context-local-pg",
                "database": "llm_context",
            },
        },
    )
    monkeypatch.setattr(
        handler,
        "_get_database_runtime_summary",
        lambda refresh=False: {
            "reachable": True,
            "database": "llm_context",
            "server_version": "16.0",
            "pgvector_available": False,
            "schema_ready": True,
            "required_tables": {
                "documents": True,
                "chunks": True,
                "chunk_embeddings": True,
                "index_runs": True,
                "symbols": True,
            },
            "error": None,
        },
    )

    payload = handler.get_operational_status(ready=True)

    assert payload["runtime_readiness"]["status"] == "blocked"
    assert "pgvector_extension_missing" in payload["runtime_readiness"]["blocking_reasons"]
    assert any(
        "supporti pgvector" in item
        for item in payload["runtime_readiness"]["recommended_actions"]
    )


def test_context_operational_status_marks_missing_schema_as_blocked(
    monkeypatch, no_warmup, tmp_path
):
    monkeypatch.setenv(
        "LLM_CONTEXT_DSN",
        "postgresql://ctx_user:ctx_pass@db-host:5432/llm_context",
    )
    monkeypatch.setenv("LLM_CONTEXT_STORE_TARGET", "llm-context-local-pg")
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T13:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T13:00:00Z",
            "index_fingerprint": "idx-alpha",
            "store_target": {
                "name": "llm-context-local-pg",
                "database": "llm_context",
            },
        },
    )
    monkeypatch.setattr(
        handler,
        "_get_database_runtime_summary",
        lambda refresh=False: {
            "reachable": True,
            "database": "llm_context",
            "server_version": "16.0",
            "pgvector_available": True,
            "schema_ready": False,
            "required_tables": {
                "documents": True,
                "chunks": True,
                "chunk_embeddings": False,
                "index_runs": True,
                "symbols": True,
            },
            "error": None,
        },
    )

    payload = handler.get_operational_status(ready=True)

    assert payload["runtime_readiness"]["status"] == "blocked"
    assert "database_schema_not_initialized" in payload["runtime_readiness"]["blocking_reasons"]
    assert any(
        "configurato nel DSN" in item
        for item in payload["runtime_readiness"]["recommended_actions"]
    )


def test_context_operational_status_marks_not_indexed_projects_as_blocked(
    monkeypatch, no_warmup, tmp_path
):
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))

    payload = handler.get_operational_status(ready=True)

    assert payload["runtime_readiness"]["status"] == "blocked"
    assert payload["runtime_readiness"]["ready_for_queries"] is False
    assert "no_project_with_integrity_ok" in payload["runtime_readiness"]["blocking_reasons"]
    assert "registered_projects_are_not_indexed" in payload["runtime_readiness"]["warnings"]


def test_storage_target_summary_marks_dedicated_database(monkeypatch, no_warmup):
    monkeypatch.setenv(
        "LLM_CONTEXT_DSN",
        "postgresql://ctx_user:ctx_pass@localhost:5432/llm_context",
    )
    monkeypatch.setenv("LLM_CONTEXT_STORE_TARGET", "llm-context-local-pg")

    handler = MCPHandler()
    payload = handler.get_operational_status(ready=True)

    assert payload["storage_target"]["name"] == "llm-context-local-pg"
    assert payload["storage_target"]["database"] == "llm_context"
    assert payload["storage_target"]["dedicated_candidate"] is True
    assert payload["storage_target"]["deployment_hint"] == "local_or_docker_port_mapping"


def test_storage_target_summary_marks_docker_alias_target(monkeypatch, no_warmup):
    monkeypatch.setenv(
        "LLM_CONTEXT_DSN",
        "postgresql://ctx_user:ctx_pass@host.docker.internal:5432/llm_context",
    )

    handler = MCPHandler()
    payload = handler.get_operational_status(ready=True)

    assert payload["storage_target"]["network_scope"] == "docker_alias"
    assert payload["storage_target"]["deployment_hint"] == "docker_network_alias"


def test_database_unreachable_localhost_suggests_local_or_docker_runbook(
    monkeypatch, no_warmup, tmp_path
):
    monkeypatch.setenv(
        "LLM_CONTEXT_DSN",
        "postgresql://ctx_user:ctx_pass@127.0.0.1:5432/llm_context",
    )
    monkeypatch.setenv("LLM_CONTEXT_STORE_TARGET", "llm-context-local-pg")
    handler = MCPHandler(config_path=str(_write_multi_project_config(tmp_path)))
    handler._project_registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-27T13:00:00Z",
        index_version="v2",
        index_fingerprint="idx-alpha",
    )
    handler._project_registry.save_index_manifest(
        "alpha",
        {
            "index_version": "v2",
            "last_ingest_status": "success",
            "last_ingest_completed_at": "2026-03-27T13:00:00Z",
            "index_fingerprint": "idx-alpha",
            "store_target": {
                "name": "llm-context-local-pg",
                "database": "llm_context",
            },
        },
    )
    monkeypatch.setattr(
        handler,
        "_get_database_runtime_summary",
        lambda refresh=False: {
            "reachable": False,
            "database": None,
            "host": "127.0.0.1",
            "port": 5432,
            "network_scope": "loopback",
            "deployment_hint": "local_or_docker_port_mapping",
            "server_version": None,
            "pgvector_available": None,
            "schema_ready": None,
            "required_tables": {},
            "error": "OperationalError: connection refused",
        },
    )

    payload = handler.get_operational_status(ready=True)

    assert any(
        "Docker con port mapping" in item
        for item in payload["runtime_readiness"]["recommended_actions"]
    )











