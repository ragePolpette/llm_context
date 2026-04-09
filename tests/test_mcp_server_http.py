import types
from logging.handlers import RotatingFileHandler

import pytest

import mcp_server_http
from mcp_server_http import _coerce_rpc_message


def test_raw_rpc_arguments_are_wrapped_for_rag_context():
    message = _coerce_rpc_message({"query_text": "legacylib", "top_k": 3})

    assert message["method"] == "tools/call"
    assert message["params"]["name"] == "rag_context"
    assert message["params"]["arguments"]["query_text"] == "legacylib"


def test_raw_rpc_arguments_reject_unknown_keys():
    try:
        _coerce_rpc_message({"query_text": "legacylib", "dsn": "postgres://evil"})
    except ValueError as exc:
        assert "Unsupported /rpc rag_context arguments" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported /rpc argument")


def test_raw_rpc_arguments_require_query_input():
    try:
        _coerce_rpc_message({"top_k": 3})
    except ValueError as exc:
        assert "requires 'query_text' or 'query_embedding'" in str(exc)
    else:
        raise AssertionError("Expected ValueError when query input is missing")


def test_jsonrpc_message_requires_valid_method_and_params():
    try:
        _coerce_rpc_message({"jsonrpc": "2.0", "method": "", "params": []})
    except ValueError as exc:
        assert "non-empty string 'method'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid JSON-RPC envelope")


def test_tools_call_requires_object_arguments():
    try:
        _coerce_rpc_message(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "rag_context", "arguments": "bad"},
            }
        )
    except ValueError as exc:
        assert "params.arguments" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid tools/call arguments")


def test_tools_call_rag_context_rejects_unknown_arguments():
    with pytest.raises(ValueError, match="Unsupported /rpc rag_context arguments"):
        _coerce_rpc_message(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "rag_context",
                    "arguments": {"query_text": "legacylib", "dsn": "evil"},
                },
            }
        )


def test_non_windows_port_owner_lookup_uses_lsof(monkeypatch):
    calls = []

    def fake_run_subprocess(command):
        calls.append(command)
        return types.SimpleNamespace(stdout="4321\n")

    monkeypatch.setattr(mcp_server_http, "_IS_WINDOWS", False)
    monkeypatch.setattr(mcp_server_http, "_run_subprocess", fake_run_subprocess)

    assert mcp_server_http._get_port_owner_pid(8765) == 4321
    assert calls == [["lsof", "-ti", "tcp:8765"]]


def test_non_windows_process_lookup_uses_ps(monkeypatch):
    calls = []

    def fake_run_subprocess(command):
        calls.append(command)
        return types.SimpleNamespace(stdout="python mcp_server_http.py")

    monkeypatch.setattr(mcp_server_http, "_IS_WINDOWS", False)
    monkeypatch.setattr(mcp_server_http, "_run_subprocess", fake_run_subprocess)

    assert mcp_server_http._is_own_server_process(321) is True
    assert calls == [["ps", "-p", "321", "-o", "command="]]


def test_non_windows_kill_does_not_call_taskkill(monkeypatch):
    calls = []
    attempts = {"count": 0}

    def fake_run_subprocess(command):
        calls.append(command)
        return types.SimpleNamespace(stdout="")

    def fake_kill(pid, signal_number):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return None
        raise OSError("process exited")

    monkeypatch.setattr(mcp_server_http, "_IS_WINDOWS", False)
    monkeypatch.setattr(mcp_server_http, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(mcp_server_http.os, "kill", fake_kill)

    assert mcp_server_http._kill_old_server(555) is True
    assert calls == []


def test_configure_http_logging_uses_rotating_file_handler(monkeypatch, tmp_path):
    log_path = tmp_path / "mcp-http.log"
    monkeypatch.setenv("LLM_CONTEXT_HTTP_LOG_PATH", str(log_path))
    monkeypatch.setenv("LLM_CONTEXT_HTTP_LOG_MAX_BYTES", "2048")
    monkeypatch.setenv("LLM_CONTEXT_HTTP_LOG_BACKUP_COUNT", "4")

    http_logger = mcp_server_http.logging.getLogger("mcp_server_http")
    original_handlers = list(http_logger.handlers)

    try:
        mcp_server_http._configure_http_logging()

        rotating = [
            handler
            for handler in http_logger.handlers
            if isinstance(handler, RotatingFileHandler)
        ]
        assert len(rotating) == 1
        assert rotating[0].baseFilename == str(log_path)
        assert rotating[0].maxBytes == 2048
        assert rotating[0].backupCount == 4
        assert any(
            isinstance(handler, mcp_server_http.logging.StreamHandler)
            and not isinstance(handler, RotatingFileHandler)
            for handler in http_logger.handlers
        )
    finally:
        for handler in list(http_logger.handlers):
            handler.close()
        http_logger.handlers.clear()
        for handler in original_handlers:
            http_logger.addHandler(handler)


def test_configure_http_logging_falls_back_on_invalid_env(monkeypatch, tmp_path):
    log_path = tmp_path / "mcp-http.log"
    monkeypatch.setenv("LLM_CONTEXT_HTTP_LOG_PATH", str(log_path))
    monkeypatch.setenv("LLM_CONTEXT_HTTP_LOG_MAX_BYTES", "invalid")
    monkeypatch.setenv("LLM_CONTEXT_HTTP_LOG_BACKUP_COUNT", "-1")

    http_logger = mcp_server_http.logging.getLogger("mcp_server_http")
    original_handlers = list(http_logger.handlers)

    try:
        mcp_server_http._configure_http_logging()
        rotating = next(
            handler
            for handler in http_logger.handlers
            if isinstance(handler, RotatingFileHandler)
        )
        assert rotating.maxBytes == mcp_server_http._DEFAULT_HTTP_LOG_MAX_BYTES
        assert rotating.backupCount == mcp_server_http._DEFAULT_HTTP_LOG_BACKUP_COUNT
    finally:
        for handler in list(http_logger.handlers):
            handler.close()
        http_logger.handlers.clear()
        for handler in original_handlers:
            http_logger.addHandler(handler)


@pytest.mark.asyncio
async def test_read_json_body_rejects_oversized_payload():
    payload = b'{"query_embedding": [' + b"1," * 400_000 + b"1]}"

    class FakeContent:
        async def iter_chunked(self, _size):
            yield payload

    class FakeRequest:
        content_length = len(payload)
        content = FakeContent()

    with pytest.raises(ValueError, match="too large"):
        await mcp_server_http._read_json_body(FakeRequest(), 1024)


def test_build_health_payload_uses_handler_operational_status():
    class FakeReady:
        def is_set(self):
            return True

    class FakeHandler:
        def __init__(self):
            self._ready = FakeReady()

        def get_operational_status(self, *, ready):
            return {"status": "ready" if ready else "loading", "project_count": 2}

    payload = mcp_server_http._build_health_payload(FakeHandler())

    assert payload == {"status": "ready", "project_count": 2}


def test_resolve_runtime_config_path_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_CONFIG_PATH", "config.rework.yaml")

    assert mcp_server_http._resolve_runtime_config_path() == "config.rework.yaml"


def test_resolve_runtime_config_path_returns_none_when_not_set(monkeypatch):
    monkeypatch.delenv("LLM_CONTEXT_CONFIG_PATH", raising=False)

    assert mcp_server_http._resolve_runtime_config_path() is None


