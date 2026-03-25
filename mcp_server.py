"""
MCP Server using stdio for persistent operation.

Simple stdio wrapper around the MCPHandler.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Force project path for local imports
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Log stderr to file for debug without breaking stdout
_ERR_LOG = _ROOT_DIR / "mcp_server.err.log"
try:
    _ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr = open(_ERR_LOG, "a", encoding="utf-8", buffering=1)
except Exception:
    pass

from rag_indexer.mcp_handler import MCPHandler


def _resolve_runtime_config_path() -> Optional[str]:
    value = str(os.getenv("LLM_CONTEXT_CONFIG_PATH", "")).strip()
    return value or None


def _read_headers() -> dict[str, str]:
    """Read HTTP-style headers from stdin."""
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line == b"":
            return {}
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8", errors="replace")
        key, _, value = decoded.partition(":")
        if key:
            headers[key.strip().lower()] = value.strip()
    return headers


def _read_message() -> Optional[dict[str, Any]]:
    """Read a JSON-RPC message from stdin."""
    header = _read_headers()
    if not header:
        return None
    length = int(header.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _send_message(message: dict[str, Any]) -> None:
    """Send a JSON-RPC message to stdout."""
    data = json.dumps(message)
    payload = data.encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
    sys.stdout.buffer.write(header + payload)
    sys.stdout.buffer.flush()


def main() -> None:
    """Main stdio loop."""
    config_path = _resolve_runtime_config_path()
    handler = MCPHandler(config_path=config_path)
    sys.stderr.write("[INFO] MCP Server (stdio) started. Waiting for messages...\n")
    while True:
        message = _read_message()
        if message is None:
            sys.stderr.write("[INFO] EOF received, shutting down.\n")
            return
        response = handler.handle_message(message)
        if response is not None:
            _send_message(response)


if __name__ == "__main__":
    main()
