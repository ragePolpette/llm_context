"""
MCP Server HTTP wrapper for persistent operation.

Run this server manually BEFORE opening your MCP client/editor.
The model will be loaded once and stay in memory.

Usage:
  python mcp_server_http.py
  llm-context serve --transport http

Then configure your MCP client to use:
  "command": "npx",
  "args": ["-y", "mcp-remote", "http://localhost:8765/mcp", "--transport", "http-only"]

Environment variables:
  MCP_HOST        - bind address (default: 127.0.0.1)
  MCP_PORT        - bind port (default: 8765)
  MCP_AUTH_TOKEN  - optional bearer token required on every endpoint;
                    mandatory when MCP_HOST is not a loopback address

Only one instance per checkout can run at a time (file lock). A running
instance is never killed automatically: stop it first.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import socket
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from aiohttp import web

_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from rag_indexer.mcp_handler import MCPHandler, UUIDEncoder, tool_rag_context  # noqa: E402

log = logging.getLogger("mcp_server_http")
_IS_WINDOWS = os.name == "nt"
_DEFAULT_HTTP_LOG_MAX_BYTES = 1_048_576
_DEFAULT_HTTP_LOG_BACKUP_COUNT = 3
_DEFAULT_MAX_REQUEST_BYTES = 1_048_576
_DEPRECATED_CLIENT_ARGUMENTS: dict[str, dict[str, Any]] = {
    "embedder": {"type": "string"},
    "local_model": {"type": "string"},
    "embedding_dim": {"type": "integer"},
}

_LOCK_PATH = _ROOT_DIR / ".mcp_server.lock"
_PID_PATH = _ROOT_DIR / ".mcp_server.pid"
_lock_file_handle = None


def _acquire_instance_lock() -> bool:
    """Acquire the single-instance file lock. Returns False if another live instance holds it."""
    global _lock_file_handle
    handle = None
    try:
        handle = open(_LOCK_PATH, "a+")  # noqa: SIM115
        handle.seek(0)
        if handle.read(1) == "":
            handle.seek(0)
            handle.write("0")
            handle.flush()
        handle.seek(0)
        _lock_file(handle)
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
    except OSError:
        if handle is not None:
            handle.close()
        return False
    _lock_file_handle = handle
    try:
        _PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    except OSError:
        pass
    return True


def _read_lock_pid() -> Optional[int]:
    """Read the PID of the running instance (best-effort)."""
    try:
        text = _PID_PATH.read_text(encoding="ascii").strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def _release_instance_lock() -> None:
    """Release the single-instance lock."""
    global _lock_file_handle
    if _lock_file_handle:
        try:
            _lock_file_handle.seek(0)
            _unlock_file(_lock_file_handle)
        except OSError:
            pass
        _lock_file_handle.close()
        _lock_file_handle = None
    for path in (_LOCK_PATH, _PID_PATH):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _ensure_port_free(host: str, port: int) -> None:
    """Raise RuntimeError if *port* is busy. Never kills the owning process."""
    if _is_port_available(host, port):
        return
    raise RuntimeError(
        f"Port {port} is already in use. Stop the process listening on it "
        f"({_manual_port_check_hint(port)}) or set MCP_PORT to a different value."
    )


def _lock_file(file_obj) -> None:
    if _IS_WINDOWS:
        import msvcrt

        msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file_obj) -> None:
    if _IS_WINDOWS:
        import msvcrt

        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def _manual_port_check_hint(port: int) -> str:
    if _IS_WINDOWS:
        return f"Get-NetTCPConnection -LocalPort {port}"
    return f"lsof -i tcp:{port}"


def _is_loopback_host(host: str) -> bool:
    normalized = _normalize_host(host)
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _resolve_auth_token() -> Optional[str]:
    token = str(os.getenv("MCP_AUTH_TOKEN", "")).strip()
    return token or None


def _validate_bind_security(host: str, auth_token: Optional[str]) -> None:
    if _is_loopback_host(host) or auth_token:
        return
    raise RuntimeError(
        f"Refusing to bind to non-loopback address {host!r} without authentication. "
        "Set MCP_AUTH_TOKEN or bind to 127.0.0.1."
    )


def _is_authorized(authorization_header: Optional[str], auth_token: Optional[str]) -> bool:
    if not auth_token:
        return True
    if not authorization_header:
        return False
    scheme, _, credentials = authorization_header.strip().partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(credentials.strip().encode("utf-8"), auth_token.encode("utf-8"))


def _parse_non_negative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        print(
            f"[WARN] Invalid {name}={raw!r}; using default {default}",
            file=sys.stderr,
        )
        return default
    if value < 0:
        print(
            f"[WARN] Invalid {name}={raw!r}; using default {default}",
            file=sys.stderr,
        )
        return default
    return value


def _configure_http_logging() -> None:
    http_logger = logging.getLogger("mcp_server_http")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    log_path = Path(
        os.getenv(
            "LLM_CONTEXT_HTTP_LOG_PATH",
            str(_ROOT_DIR / "logs" / "mcp_server_http.log"),
        )
    ).expanduser()
    if not log_path.is_absolute():
        log_path = (_ROOT_DIR / log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = _parse_non_negative_int_env(
        "LLM_CONTEXT_HTTP_LOG_MAX_BYTES",
        _DEFAULT_HTTP_LOG_MAX_BYTES,
    )
    backup_count = _parse_non_negative_int_env(
        "LLM_CONTEXT_HTTP_LOG_BACKUP_COUNT",
        _DEFAULT_HTTP_LOG_BACKUP_COUNT,
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    for handler in list(http_logger.handlers):
        handler.close()
        http_logger.removeHandler(handler)
    http_logger.propagate = False
    http_logger.setLevel(logging.INFO)
    http_logger.addHandler(stream_handler)
    http_logger.addHandler(file_handler)


# ============================================================================
# HTTP/SSE Server using aiohttp
# ============================================================================

def _parse_csv_env(raw_value: Optional[str], default: list[str]) -> list[str]:
    if raw_value is None:
        return list(default)
    parts = [part.strip() for part in raw_value.split(",")]
    values = [part for part in parts if part]
    return values or list(default)


def _parse_bool_env(raw_value: Optional[str], default: bool) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_host(value: str) -> str:
    host = value.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _normalize_request_host(host_header: Optional[str]) -> Optional[str]:
    if not host_header:
        return None
    host = host_header.strip()
    if not host:
        return None
    parsed = urlsplit(f"http://{host}")
    if not parsed.hostname:
        return None
    return _normalize_host(parsed.hostname)


def _normalize_allowed_host(entry: str) -> str:
    candidate = entry.strip().lower()
    if not candidate:
        return ""
    if "://" in candidate:
        parsed = urlsplit(candidate)
        if parsed.hostname:
            return _normalize_host(parsed.hostname)
    if ":" in candidate and not candidate.endswith(":*"):
        parsed = urlsplit(f"http://{candidate}")
        if parsed.hostname:
            return _normalize_host(parsed.hostname)
    if candidate.endswith(":*"):
        candidate = candidate[:-2]
    return _normalize_host(candidate)


def _is_host_allowed(host_header: Optional[str], allowed_hosts: list[str]) -> bool:
    request_host = _normalize_request_host(host_header)
    if not request_host:
        return False
    normalized = {_normalize_allowed_host(entry) for entry in allowed_hosts}
    return request_host in normalized


def _normalize_origin(origin: str) -> Optional[str]:
    try:
        parsed = urlsplit(origin)
    except Exception:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    base = f"{parsed.scheme.lower()}://{_normalize_host(parsed.hostname)}"
    if parsed.port is not None:
        return f"{base}:{parsed.port}"
    return base


def _is_origin_allowed(origin: str, allowed_origins: list[str]) -> bool:
    normalized_origin = _normalize_origin(origin)
    if not normalized_origin:
        return False

    for raw in allowed_origins:
        allowed = raw.strip().lower()
        if not allowed:
            continue
        if allowed == normalized_origin:
            return True
        if allowed.endswith(":*"):
            prefix = allowed[:-2]
            if normalized_origin.startswith(prefix + ":"):
                return True
    return False


def _jsonrpc_transport_error(message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "error": {"code": -32000, "message": message},
        "id": None,
    }


def _build_health_payload(handler: MCPHandler) -> dict[str, Any]:
    payload = handler.get_operational_status(ready=handler._ready.is_set())
    # Enrich with cache stats
    with handler._query_cache_lock:
        cache_entries = len(handler._query_cache)
    payload["query_cache"] = {
        "entries": cache_entries,
        "max_entries": handler._query_cache_max,
        "ttl_seconds": handler._query_cache_ttl,
    }
    return payload


def _resolve_runtime_config_path() -> Optional[str]:
    value = str(os.getenv("LLM_CONTEXT_CONFIG_PATH", "")).strip()
    return value or None


def _coerce_rpc_message(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("/rpc expects a JSON object body")
    if "jsonrpc" not in body and "method" not in body:
        return _wrap_raw_rpc_arguments(body)
    _validate_jsonrpc_message(body)
    return body


def _validate_jsonrpc_message(body: dict[str, Any]) -> None:
    method = body.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("JSON-RPC request must include a non-empty string 'method'")

    params = body.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError("JSON-RPC 'params' must be an object when provided")

    if method == "tools/call" and isinstance(params, dict):
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tools/call requires a non-empty string 'params.name'")
        if not isinstance(arguments, dict):
            raise ValueError("tools/call requires 'params.arguments' to be an object")
        if name == "rag_context":
            _validate_raw_rag_context_arguments(arguments)


def _wrap_raw_rpc_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    validated = _validate_raw_rag_context_arguments(arguments)
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "rag_context",
            "arguments": validated,
        },
    }


def _validate_raw_rag_context_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    schema_properties = {
        **_DEPRECATED_CLIENT_ARGUMENTS,
        **tool_rag_context()["inputSchema"]["properties"],
    }
    unknown_keys = sorted(set(arguments) - set(schema_properties))
    if unknown_keys:
        raise ValueError(
            "Unsupported /rpc rag_context arguments: " + ", ".join(unknown_keys)
        )
    if "query_text" not in arguments and "query_embedding" not in arguments:
        raise ValueError("/rpc rag_context requires 'query_text' or 'query_embedding'")

    for key, value in arguments.items():
        _validate_argument_value(key, value, schema_properties[key])
    return dict(arguments)


def _validate_argument_value(name: str, value: Any, schema: dict[str, Any]) -> None:
    schema_type = schema.get("type")
    if schema_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"/rpc argument '{name}' must be a string")
        return
    if schema_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"/rpc argument '{name}' must be an integer")
        return
    if schema_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"/rpc argument '{name}' must be a boolean")
        return
    if schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"/rpc argument '{name}' must be an array")
        item_schema = schema.get("items", {})
        if item_schema.get("type") == "number":
            for item in value:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError(
                        f"/rpc argument '{name}' must contain only numeric values"
                    )
        return


async def _read_json_body(request: web.Request, max_bytes: int) -> Any:
    content_length = request.content_length
    if content_length is not None and content_length > max_bytes:
        raise ValueError(
            f"Request body too large ({content_length} > {max_bytes} bytes)"
        )

    body = bytearray()
    async for chunk in request.content.iter_chunked(65_536):
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ValueError(
                f"Request body too large ({len(body)} > {max_bytes} bytes)"
            )

    if not body:
        raise ValueError("Empty request body")

    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON") from exc


def build_app(
    handler: MCPHandler,
    *,
    sse_enabled: bool = False,
    allowed_hosts: Optional[list[str]] = None,
    allowed_origins: Optional[list[str]] = None,
    auth_token: Optional[str] = None,
    max_request_bytes: Optional[int] = None,
) -> web.Application:
    """Build the aiohttp application exposing the MCP transports."""
    if allowed_hosts is None:
        allowed_hosts = ["localhost", "127.0.0.1", "::1"]
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:*",
            "http://127.0.0.1:*",
            "https://localhost:*",
            "https://127.0.0.1:*",
        ]
    if max_request_bytes is None:
        max_request_bytes = _parse_non_negative_int_env(
            "LLM_CONTEXT_MAX_REQUEST_BYTES",
            _DEFAULT_MAX_REQUEST_BYTES,
        )

    sessions: dict[str, asyncio.Queue] = {}

    async def dispatch(body: Any) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(handler.handle_message, body)

    @web.middleware
    async def transport_security_middleware(request: web.Request, handler):
        if not _is_authorized(request.headers.get("Authorization"), auth_token):
            return web.json_response(
                _jsonrpc_transport_error("Unauthorized"),
                status=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if request.path != "/health":
            host_header = request.headers.get("Host")
            if not _is_host_allowed(host_header, allowed_hosts):
                return web.json_response(
                    _jsonrpc_transport_error("Forbidden: invalid Host header"),
                    status=403,
                )

            origin = request.headers.get("Origin")
            if origin and not _is_origin_allowed(origin, allowed_origins):
                return web.json_response(
                    _jsonrpc_transport_error("Forbidden: Origin not allowed"),
                    status=403,
                )
        return await handler(request)

    async def handle_sse(request: web.Request) -> web.StreamResponse:
        """SSE endpoint for MCP protocol."""
        session_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        sessions[session_id] = queue

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        origin = request.headers.get("Origin")
        if origin and _is_origin_allowed(origin, allowed_origins):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Vary"] = "Origin"

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers=headers,
        )
        await response.prepare(request)

        endpoint_url = f"{request.scheme}://{request.host}/message?session_id={session_id}"
        await response.write(f"event: endpoint\ndata: {endpoint_url}\n\n".encode())

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    data = json.dumps(msg, cls=UUIDEncoder)
                    await response.write(f"event: message\ndata: {data}\n\n".encode())
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            sessions.pop(session_id, None)

        return response

    async def handle_message(request: web.Request) -> web.Response:
        """Handle incoming JSON-RPC messages for an SSE session."""
        session_id = request.query.get("session_id")
        if not session_id or session_id not in sessions:
            return web.json_response({"error": "Invalid session"}, status=400)

        try:
            body = await _read_json_body(request, max_request_bytes)
        except ValueError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        response = await dispatch(body)
        queue = sessions.get(session_id)
        if response is not None and queue is not None:
            await queue.put(response)

        return web.Response(status=202, text="Accepted")

    async def handle_health(request: web.Request) -> web.Response:
        """Health check endpoint."""
        payload = await asyncio.to_thread(_build_health_payload, handler)
        return web.json_response(payload, dumps=lambda obj: json.dumps(obj, cls=UUIDEncoder))

    async def handle_rpc(request: web.Request) -> web.Response:
        """Handle synchronous JSON-RPC messages (no SSE)."""
        try:
            body = await _read_json_body(request, max_request_bytes)
            body = _coerce_rpc_message(body)
        except ValueError as e:
            log.debug("Rejected /rpc request: %s", e)
            status = 413 if "too large" in str(e).lower() else 400
            return web.json_response({"error": str(e)}, status=status)

        log.debug("/rpc method=%s id=%s", body.get("method"), body.get("id"))
        response = await dispatch(body)
        if response is None:
            return web.Response(status=204)

        result_json = json.dumps(response, cls=UUIDEncoder)
        log.debug("/rpc response bytes=%d", len(result_json))
        return web.Response(text=result_json, content_type="application/json")

    async def handle_streamable(request: web.Request) -> web.Response:
        """Handle streamable HTTP JSON-RPC requests (POST)."""
        try:
            body = await _read_json_body(request, max_request_bytes)
        except ValueError as exc:
            if "too large" in str(exc).lower():
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32000, "message": str(exc)},
                    },
                    status=413,
                )
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                },
                status=400,
            )

        response = await dispatch(body)
        if response is None:
            return web.Response(status=202, text="Accepted")

        return web.Response(
            text=json.dumps(response, cls=UUIDEncoder),
            content_type="application/json",
        )

    async def handle_streamable_delete(request: web.Request) -> web.Response:
        """Handle streamable HTTP session close requests (DELETE)."""
        return web.Response(status=200, text="OK")

    async def handle_mcp_get(request: web.Request) -> web.StreamResponse:
        """GET /mcp is either SSE (optional) or 405 (spec-compliant fallback)."""
        if sse_enabled:
            return await handle_sse(request)
        return web.Response(status=405, text="Method Not Allowed", headers={"Allow": "POST, DELETE"})

    app = web.Application(middlewares=[transport_security_middleware])
    app.router.add_get("/sse", handle_sse)
    app.router.add_post("/message", handle_message)
    app.router.add_post("/sse", handle_streamable)
    app.router.add_delete("/sse", handle_streamable_delete)
    app.router.add_get("/mcp", handle_mcp_get)
    app.router.add_post("/mcp", handle_streamable)
    app.router.add_delete("/mcp", handle_streamable_delete)
    app.router.add_post("/rpc", handle_rpc)
    app.router.add_get("/health", handle_health)
    return app


async def run_http_server(
    handler: MCPHandler,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    sse_enabled: bool = False,
    allowed_hosts: Optional[list[str]] = None,
    allowed_origins: Optional[list[str]] = None,
    auth_token: Optional[str] = None,
):
    """Run the HTTP server until cancelled."""
    _validate_bind_security(host, auth_token)
    app = build_app(
        handler,
        sse_enabled=sse_enabled,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        auth_token=auth_token,
    )
    runner = web.AppRunner(app)
    await runner.setup()

    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
    except OSError as exc:
        log.error(
            "Failed to bind %s:%d (%s). Check with: %s",
            host,
            port,
            exc,
            _manual_port_check_hint(port),
        )
        await runner.cleanup()
        raise

    log.info("MCP HTTP server running at http://%s:%d/mcp", host, port)
    log.info("MCP SSE legacy endpoint at http://%s:%d/sse", host, port)
    log.info("Health check: http://%s:%d/health", host, port)
    log.info("Bearer authentication: %s", "enabled" if auth_token else "disabled")
    log.info("Press Ctrl+C to stop")

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def main():
    _configure_http_logging()

    log.info("=" * 50)
    log.info("  LLM Context MCP Server (HTTP/SSE)")
    log.info("=" * 50)

    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8765"))
    auth_token = _resolve_auth_token()
    try:
        _validate_bind_security(host, auth_token)
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if not _acquire_instance_lock():
        log.error(
            "Another instance is already running (PID %s, lock file: %s). Stop it first.",
            _read_lock_pid() or "unknown",
            _LOCK_PATH,
        )
        sys.exit(1)

    sse_enabled = _parse_bool_env(os.getenv("MCP_SSE_ENABLED"), False)
    allowed_hosts = _parse_csv_env(
        os.getenv("MCP_ALLOWED_HOSTS"),
        ["localhost", "127.0.0.1", "::1"],
    )
    allowed_origins = _parse_csv_env(
        os.getenv("MCP_ALLOWED_ORIGINS"),
        [
            "http://localhost:*",
            "http://127.0.0.1:*",
            "https://localhost:*",
            "https://127.0.0.1:*",
        ],
    )

    try:
        _ensure_port_free(host, port)
    except RuntimeError as exc:
        log.error("%s", exc)
        _release_instance_lock()
        sys.exit(1)

    config_path = _resolve_runtime_config_path()
    if config_path:
        log.info("Using runtime config: %s", config_path)
    else:
        log.info("Using default runtime config: config.yaml")

    handler = MCPHandler(config_path=config_path)

    # Wait for warmup before accepting connections
    log.info("Waiting for embedder to load...")
    try:
        if handler.wait_ready(timeout=120.0):
            log.info("Embedder ready!")
        else:
            log.warning("Embedder warmup timed out, continuing anyway...")
    except KeyboardInterrupt:
        log.warning("Warmup interrupted, continuing anyway...")

    try:
        log.info("MCP transport security: hosts=%s origins=%s", allowed_hosts, allowed_origins)
        log.info("MCP GET /mcp SSE enabled: %s", sse_enabled)
        asyncio.run(
            run_http_server(
                handler,
                host=host,
                port=port,
                sse_enabled=sse_enabled,
                allowed_hosts=allowed_hosts,
                allowed_origins=allowed_origins,
                auth_token=auth_token,
            )
        )
    except KeyboardInterrupt:
        log.info("Server stopped.")
    finally:
        _release_instance_lock()


if __name__ == "__main__":
    main()
