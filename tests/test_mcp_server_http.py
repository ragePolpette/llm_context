import asyncio
import threading
import time
from logging.handlers import RotatingFileHandler

import pytest
from aiohttp.test_utils import TestClient, TestServer

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
            self._query_cache_lock = threading.Lock()
            self._query_cache = {}
            self._query_cache_max = 8
            self._query_cache_ttl = 60

        def get_operational_status(self, *, ready):
            return {"status": "ready" if ready else "loading", "project_count": 2}

    payload = mcp_server_http._build_health_payload(FakeHandler())

    assert payload["status"] == "ready"
    assert payload["project_count"] == 2
    assert payload["query_cache"] == {"entries": 0, "max_entries": 8, "ttl_seconds": 60}


def test_resolve_runtime_config_path_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_CONFIG_PATH", "config.rework.yaml")

    assert mcp_server_http._resolve_runtime_config_path() == "config.rework.yaml"


def test_resolve_runtime_config_path_returns_none_when_not_set(monkeypatch):
    monkeypatch.delenv("LLM_CONTEXT_CONFIG_PATH", raising=False)

    assert mcp_server_http._resolve_runtime_config_path() is None



class _SlowHandler:
    def __init__(self, delay=0.4):
        self.delay = delay
        self.threads = []
        self._ready = threading.Event()
        self._ready.set()
        self._query_cache_lock = threading.Lock()
        self._query_cache = {}
        self._query_cache_max = 1
        self._query_cache_ttl = 1

    def handle_message(self, message):
        self.threads.append(threading.current_thread())
        time.sleep(self.delay)
        return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"ok": True}}

    def get_operational_status(self, *, ready):
        self.threads.append(threading.current_thread())
        return {"status": "ready"}


async def _client_for(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_block_event_loop():
    handler = _SlowHandler(delay=0.4)
    client = await _client_for(mcp_server_http.build_app(handler))
    try:
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        started = time.monotonic()
        responses = await asyncio.gather(
            client.post("/mcp", json=body),
            client.post("/rpc", json=body),
        )
        elapsed = time.monotonic() - started
        assert [response.status for response in responses] == [200, 200]
    finally:
        await client.close()

    assert elapsed < 0.75
    assert all(thread is not threading.main_thread() for thread in handler.threads)


@pytest.mark.asyncio
async def test_health_runs_status_probe_off_the_event_loop():
    handler = _SlowHandler()
    client = await _client_for(mcp_server_http.build_app(handler))
    try:
        response = await client.get("/health")
        assert response.status == 200
    finally:
        await client.close()

    assert handler.threads and handler.threads[0] is not threading.main_thread()


@pytest.mark.asyncio
async def test_bearer_token_is_required_when_configured():
    handler = _SlowHandler(delay=0)
    app = mcp_server_http.build_app(handler, auth_token="s3cret")
    client = await _client_for(app)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    try:
        missing = await client.post("/mcp", json=body)
        wrong = await client.post("/mcp", json=body, headers={"Authorization": "Bearer nope"})
        health = await client.get("/health")
        ok = await client.post("/mcp", json=body, headers={"Authorization": "Bearer s3cret"})
        assert missing.status == 401
        assert missing.headers["WWW-Authenticate"] == "Bearer"
        assert wrong.status == 401
        assert health.status == 401
        assert ok.status == 200
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_no_token_configured_keeps_local_access_open():
    client = await _client_for(mcp_server_http.build_app(_SlowHandler(delay=0)))
    try:
        response = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert response.status == 200
    finally:
        await client.close()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_non_loopback_bind_requires_token(host):
    with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN"):
        mcp_server_http._validate_bind_security(host, None)
    mcp_server_http._validate_bind_security(host, "token")


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"])
def test_loopback_bind_does_not_require_token(host):
    mcp_server_http._validate_bind_security(host, None)


def test_resolve_auth_token_reads_env(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", "  abc  ")
    assert mcp_server_http._resolve_auth_token() == "abc"
    monkeypatch.setenv("MCP_AUTH_TOKEN", "")
    assert mcp_server_http._resolve_auth_token() is None


def test_instance_lock_held_elsewhere_is_not_killed(monkeypatch, tmp_path):
    lock_path = tmp_path / ".mcp_server.lock"
    pid_path = tmp_path / ".mcp_server.pid"
    monkeypatch.setattr(mcp_server_http, "_LOCK_PATH", lock_path)
    monkeypatch.setattr(mcp_server_http, "_PID_PATH", pid_path)
    killed = []
    monkeypatch.setattr(mcp_server_http.os, "kill", lambda *args: killed.append(args))

    with open(lock_path, "a+") as holder:
        mcp_server_http._lock_file(holder)
        try:
            assert mcp_server_http._acquire_instance_lock() is False
        finally:
            mcp_server_http._unlock_file(holder)

    assert killed == []
    assert mcp_server_http._acquire_instance_lock() is True
    assert mcp_server_http._read_lock_pid() == mcp_server_http.os.getpid()
    mcp_server_http._release_instance_lock()
    assert not lock_path.exists()
    assert not pid_path.exists()


def test_ensure_port_free_never_kills(monkeypatch):
    monkeypatch.setattr(mcp_server_http, "_is_port_available", lambda host, port: False)
    with pytest.raises(RuntimeError, match="already in use"):
        mcp_server_http._ensure_port_free("127.0.0.1", 8765)


def test_raw_rpc_forwards_deprecated_embedder_arguments_to_handler():
    message = _coerce_rpc_message({"query_text": "q", "embedder": "local-st"})

    assert message["params"]["arguments"]["embedder"] == "local-st"
