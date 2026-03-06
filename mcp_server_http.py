"""
MCP Server HTTP/SSE wrapper for persistent operation.

Run this server manually BEFORE opening your MCP client/editor.
The model will be loaded once and stay in memory.

Usage:
  python mcp_server_http.py

Then configure mcp_config.json to use:
  "command": "npx",
  "args": ["-y", "mcp-remote", "http://localhost:8765/mcp", "--transport", "http-only"]

Environment variables:
  MCP_HOST  - bind address (default: 127.0.0.1)
  MCP_PORT  - bind port (default: 8765)

If the port is occupied by a previous instance of this server,
it will be killed automatically before binding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit
import uuid

# Force project path for local imports
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from rag_indexer.mcp_handler import MCPHandler, UUIDEncoder, tool_rag_context

log = logging.getLogger("mcp_server_http")
_IS_WINDOWS = os.name == "nt"

# ---------------------------------------------------------------------------
# Single-instance guard (file lock)
# ---------------------------------------------------------------------------
_LOCK_PATH = _ROOT_DIR / ".mcp_server.lock"
_lock_file_handle = None  # kept open for process lifetime


def _acquire_instance_lock() -> bool:
    """Try to acquire a file-based single-instance lock.
    If a stale lock exists from an old server, kill it and retry."""
    global _lock_file_handle

    def _try_lock() -> bool:
        global _lock_file_handle
        try:
            _lock_file_handle = open(_LOCK_PATH, "a+")
            _lock_file_handle.seek(0)
            first_byte = _lock_file_handle.read(1)
            if first_byte == "":
                _lock_file_handle.seek(0)
                _lock_file_handle.write("0")
                _lock_file_handle.flush()
            _lock_file_handle.seek(0)
            _lock_file(_lock_file_handle)
            _lock_file_handle.seek(0)
            _lock_file_handle.truncate()
            _lock_file_handle.write(str(os.getpid()))
            _lock_file_handle.flush()
            return True
        except (OSError, IOError):
            if _lock_file_handle:
                _lock_file_handle.close()
                _lock_file_handle = None
            return False

    if _try_lock():
        return True

    # Lock held by someone else — read old PID and try to kill it
    old_pid = _read_lock_pid()
    if old_pid is not None and old_pid != os.getpid():
        log.warning("Lock held by old PID %d, killing it...", old_pid)
        _kill_old_server(old_pid)
        time.sleep(1)
    elif old_pid is None:
        # Stale/empty lock file left after abnormal termination.
        try:
            _LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    # Retry after killing
    return _try_lock()


def _read_lock_pid() -> Optional[int]:
    """Read the PID from the lock file (best-effort)."""
    try:
        text = _LOCK_PATH.read_text().strip()
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
        except (OSError, IOError):
            pass
        _lock_file_handle.close()
        _lock_file_handle = None
    try:
        _LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Port utilities
# ---------------------------------------------------------------------------

def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _get_port_owner_pid(port: int) -> Optional[int]:
    """Return the PID listening on *port*, or None."""
    try:
        if _IS_WINDOWS:
            result = _run_subprocess(["netstat", "-ano"])
            lines = result.stdout.splitlines()
            for line in lines:
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        return int(parts[-1])
        else:
            result = _run_subprocess(["lsof", "-ti", f"tcp:{port}"])
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    return int(line)
    except Exception:
        pass
    return None


def _is_own_server_process(pid: int) -> bool:
    """Check if *pid* is a Python process running this same server script."""
    try:
        cmdline = _get_process_command(pid).lower()
        return "python" in cmdline and "mcp_server_http" in cmdline
    except Exception:
        return False


def _kill_old_server(pid: int) -> bool:
    """Kill process *pid* and wait up to 5s for it to die. Returns True if killed."""
    import signal
    log.warning("Killing old server process PID %d ...", pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        if _IS_WINDOWS:
            # process already dead or access denied — try taskkill as fallback
            try:
                _run_subprocess(["taskkill", "/F", "/PID", str(pid)])
            except Exception as exc:
                log.error("Failed to kill PID %d: %s", pid, exc)
                return False
        else:
            return False

    # Wait for process to actually exit
    for _ in range(10):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)  # 0 = just check if alive
        except OSError:
            log.info("Old server PID %d terminated.", pid)
            return True
    log.error("PID %d did not die within 5 seconds.", pid)
    return False


def _ensure_port_free(host: str, port: int) -> None:
    """Make sure *port* is free. If occupied by a previous instance of this
    server, kill it automatically. Raises RuntimeError otherwise."""
    if _is_port_available(host, port):
        return

    owner_pid = _get_port_owner_pid(port)
    if owner_pid is None:
        raise RuntimeError(
            f"Port {port} is busy but cannot identify the owning process. "
            f"Check manually: Get-NetTCPConnection -LocalPort {port}"
        )

    # Safety: only kill if it's our own old server
    if not _is_own_server_process(owner_pid):
        raise RuntimeError(
            f"Port {port} is occupied by PID {owner_pid} which is NOT "
            f"an mcp_server_http process. Will not kill it. "
            f"Free the port manually or set MCP_PORT to a different value."
        )

    if not _kill_old_server(owner_pid):
        raise RuntimeError(
            f"Could not kill old server PID {owner_pid}. "
            f"Kill it manually: taskkill /F /PID {owner_pid}"
        )

    # Wait a bit more for the port to be fully released
    for _ in range(6):
        if _is_port_available(host, port):
            return
        time.sleep(0.5)

    raise RuntimeError(
        f"Port {port} still busy after killing PID {owner_pid}. "
        f"It may be in TIME_WAIT state — wait 30s or use a different port."
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


def _run_subprocess(command: list[str]):
    import subprocess

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def _get_process_command(pid: int) -> str:
    if _IS_WINDOWS:
        result = _run_subprocess(
            [
                "wmic",
                "process",
                "where",
                f"ProcessId={pid}",
                "get",
                "CommandLine",
                "/format:list",
            ]
        )
        return result.stdout

    result = _run_subprocess(["ps", "-p", str(pid), "-o", "command="])
    return result.stdout


def _configure_http_logging() -> None:
    root_logger = logging.getLogger()
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

    max_bytes = int(os.getenv("LLM_CONTEXT_HTTP_LOG_MAX_BYTES", str(1_048_576)))
    backup_count = int(os.getenv("LLM_CONTEXT_HTTP_LOG_BACKUP_COUNT", "3"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


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
    schema_properties = tool_rag_context()["inputSchema"]["properties"]
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


async def run_http_server(
    handler: MCPHandler,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    sse_enabled: bool = False,
    allowed_hosts: Optional[list[str]] = None,
    allowed_origins: Optional[list[str]] = None,
):
    """Run HTTP server with SSE endpoint for MCP."""
    try:
        from aiohttp import web
    except ImportError:
        print("[ERROR] aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    if allowed_hosts is None:
        allowed_hosts = ["localhost", "127.0.0.1", "::1"]
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:*",
            "http://127.0.0.1:*",
            "https://localhost:*",
            "https://127.0.0.1:*",
        ]

    # Store active SSE connections
    sessions: dict[str, asyncio.Queue] = {}

    @web.middleware
    async def transport_security_middleware(request: web.Request, handler):
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

        # Send endpoint info
        endpoint_url = f"{request.scheme}://{request.host}/message?session_id={session_id}"
        await response.write(f"event: endpoint\ndata: {endpoint_url}\n\n".encode())

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    await response.write(f"event: message\ndata: {json.dumps(msg)}\n\n".encode())
                except asyncio.TimeoutError:
                    # Send keepalive
                    await response.write(b": keepalive\n\n")
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            sessions.pop(session_id, None)

        return response

    async def handle_message(request: web.Request) -> web.Response:
        """Handle incoming JSON-RPC messages."""
        session_id = request.query.get("session_id")
        if not session_id or session_id not in sessions:
            return web.json_response({"error": "Invalid session"}, status=400)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Process message
        response = handler.handle_message(body)
        if response is not None:
            await sessions[session_id].put(response)

        return web.Response(status=202, text="Accepted")

    async def handle_health(request: web.Request) -> web.Response:
        """Health check endpoint."""
        ready = handler._ready.is_set()
        return web.json_response({"status": "ready" if ready else "loading"})

    async def handle_rpc(request: web.Request) -> web.Response:
        """Handle synchronous JSON-RPC messages (no SSE)."""
        try:
            body = await request.json()
            body = _coerce_rpc_message(body)
            print(f"[RPC] Received: {json.dumps(body)}")
        except Exception as e:
            print(f"[RPC] JSON parse error: {e}")
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Process message synchronously
        response = handler.handle_message(body)
        print(f"[RPC] Response type: {type(response)}, is None: {response is None}")
        
        # If response is None (e.g. notifications), return empty success
        if response is None:
            return web.Response(status=204) # No Content
            
        # Return the response as JSON (already serializable with UUIDEncoder)
        result_json = json.dumps(response, cls=UUIDEncoder)
        print(f"[RPC] Sending response ({len(result_json)} bytes)")
        return web.Response(
            text=result_json,
            content_type='application/json'
        )

    async def handle_streamable(request: web.Request) -> web.Response:
        """Handle streamable HTTP JSON-RPC requests (POST)."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                },
                status=400,
            )

        response = handler.handle_message(body)
        if response is None:
            return web.Response(status=202, text="Accepted")

        return web.Response(
            text=json.dumps(response, cls=UUIDEncoder),
            content_type="application/json",
        )

    async def handle_streamable_delete(request: web.Request) -> web.Response:
        """Handle streamable HTTP session close requests (DELETE)."""
        return web.Response(status=200, text="OK")

    async def handle_mcp_get(request: web.Request) -> web.StreamResponse | web.Response:
        """GET /mcp is either SSE (optional) or 405 (spec-compliant fallback)."""
        if sse_enabled:
            return await handle_sse(request)
        return web.Response(status=405, text="Method Not Allowed", headers={"Allow": "POST, DELETE"})

    app = web.Application(middlewares=[transport_security_middleware])
    app.router.add_get("/sse", handle_sse)
    app.router.add_post("/message", handle_message)
    app.router.add_post("/sse", handle_streamable)  # compatibility for http-first on /sse
    app.router.add_delete("/sse", handle_streamable_delete)
    app.router.add_get("/mcp", handle_mcp_get)
    app.router.add_post("/mcp", handle_streamable)
    app.router.add_delete("/mcp", handle_streamable_delete)
    app.router.add_post("/rpc", handle_rpc)  # New synchronous endpoint
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()

    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
    except OSError as exc:
        log.error(
            "Failed to bind %s:%d — %s. "
            "Check with: Get-NetTCPConnection -LocalPort %d",
            host, port, exc, port,
        )
        await runner.cleanup()
        raise

    log.info("MCP HTTP server running at http://%s:%d/mcp", host, port)
    log.info("MCP SSE legacy endpoint at http://%s:%d/sse", host, port)
    log.info("Health check: http://%s:%d/health", host, port)
    log.info("Press Ctrl+C to stop")

    # Wait forever
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

    # --- Single-instance guard ---
    if not _acquire_instance_lock():
        log.error(
            "Another instance is already running (lock file: %s). "
            "Kill it first or delete the lock file if stale.", _LOCK_PATH
        )
        sys.exit(1)

    # --- Resolve host/port from env ---
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8765"))
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

    handler = MCPHandler()

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
            )
        )
    except KeyboardInterrupt:
        log.info("Server stopped.")
    finally:
        _release_instance_lock()


if __name__ == "__main__":
    main()
