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
from pathlib import Path
from typing import Any, Optional
import uuid

# Force project path for local imports
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from rag_indexer.mcp_handler import MCPHandler, UUIDEncoder

log = logging.getLogger("mcp_server_http")

# ---------------------------------------------------------------------------
# Single-instance guard (file lock)
# ---------------------------------------------------------------------------
_LOCK_PATH = _ROOT_DIR / ".mcp_server.lock"
_lock_file_handle = None  # kept open for process lifetime


def _acquire_instance_lock() -> bool:
    """Try to acquire a file-based single-instance lock.
    If a stale lock exists from an old server, kill it and retry."""
    global _lock_file_handle
    import msvcrt

    def _try_lock() -> bool:
        global _lock_file_handle
        try:
            _lock_file_handle = open(_LOCK_PATH, "w")
            msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
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
        import msvcrt
        try:
            _lock_file_handle.seek(0)
            msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1)
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
        import subprocess
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    return int(parts[-1])
    except Exception:
        pass
    return None


def _is_own_server_process(pid: int) -> bool:
    """Check if *pid* is a Python process running this same server script."""
    try:
        import subprocess
        result = subprocess.run(
            ["wmic", "process", "where", f"ProcessId={pid}",
             "get", "CommandLine", "/format:list"],
            capture_output=True, text=True, timeout=5,
        )
        cmdline = result.stdout.lower()
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
        # process already dead or access denied — try taskkill as fallback
        try:
            import subprocess
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=5,
            )
        except Exception as exc:
            log.error("Failed to kill PID %d: %s", pid, exc)
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


# ============================================================================
# HTTP/SSE Server using aiohttp
# ============================================================================

async def run_http_server(handler: MCPHandler, host: str = "127.0.0.1", port: int = 8765):
    """Run HTTP server with SSE endpoint for MCP."""
    try:
        from aiohttp import web
    except ImportError:
        print("[ERROR] aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    # Store active SSE connections
    sessions: dict[str, asyncio.Queue] = {}

    async def handle_sse(request: web.Request) -> web.StreamResponse:
        """SSE endpoint for MCP protocol."""
        session_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        sessions[session_id] = queue

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
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
            print(f"[RPC] Received: {json.dumps(body)}")
        except Exception as e:
            print(f"[RPC] JSON parse error: {e}")
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # If this is NOT a JSON-RPC message, wrap it as a rag_context call
        if "jsonrpc" not in body and "method" not in body:
            print(f"[RPC] Wrapping raw arguments into JSON-RPC tools/call")
            body = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "rag_context",
                    "arguments": body
                }
            }
            print(f"[RPC] Wrapped message: {json.dumps(body)}")

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

    app = web.Application()
    app.router.add_get("/sse", handle_sse)
    app.router.add_post("/message", handle_message)
    app.router.add_post("/sse", handle_streamable)  # compatibility for http-first on /sse
    app.router.add_delete("/sse", handle_streamable_delete)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
        asyncio.run(run_http_server(handler, host=host, port=port))
    except KeyboardInterrupt:
        log.info("Server stopped.")
    finally:
        _release_instance_lock()


if __name__ == "__main__":
    main()
