"""
MCP Server using stdio for persistent operation.

Implements the MCP stdio transport: one JSON-RPC message per line
(newline-delimited JSON, UTF-8). The legacy LSP-style Content-Length framing
is still available with --legacy-content-length or
MCP_STDIO_FRAMING=content-length.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO, Optional

_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from rag_indexer.mcp_handler import MCPHandler, UUIDEncoder  # noqa: E402

FRAMING_NEWLINE = "newline"
FRAMING_CONTENT_LENGTH = "content-length"


def _redirect_stderr_to_log() -> None:
    """Keep stdout reserved for protocol messages and persist diagnostics to a file."""
    log_path = Path(
        os.getenv("LLM_CONTEXT_STDIO_LOG_PATH", str(_ROOT_DIR / "mcp_server.err.log"))
    ).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    except OSError:
        pass


def _resolve_runtime_config_path() -> Optional[str]:
    value = str(os.getenv("LLM_CONTEXT_CONFIG_PATH", "")).strip()
    return value or None


def _parse_error(detail: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": f"Parse error: {detail}"},
    }


def read_newline_message(stream: BinaryIO) -> Optional[Any]:
    """Read one newline-delimited JSON-RPC message. Returns None on EOF.

    Blank lines are skipped. Malformed lines yield a JSON-RPC parse error
    payload (a dict with an "error" key and no "method") instead of raising.
    """
    while True:
        line = stream.readline()
        if line == b"":
            return None
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _parse_error(str(exc))


def _read_headers(stream: BinaryIO) -> Optional[dict[str, str]]:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            if headers:
                return headers
            continue
        decoded = line.decode("utf-8", errors="replace")
        key, _, value = decoded.partition(":")
        if key:
            headers[key.strip().lower()] = value.strip()


def read_content_length_message(stream: BinaryIO) -> Optional[Any]:
    """Read one Content-Length framed JSON-RPC message. Returns None on EOF."""
    headers = _read_headers(stream)
    if headers is None:
        return None
    try:
        length = int(headers.get("content-length", "0"))
    except ValueError:
        return _parse_error("invalid Content-Length header")
    if length <= 0:
        return _parse_error("missing Content-Length header")
    body = stream.read(length)
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _parse_error(str(exc))


def encode_message(message: dict[str, Any], framing: str = FRAMING_NEWLINE) -> bytes:
    payload = json.dumps(message, cls=UUIDEncoder, ensure_ascii=False).encode("utf-8")
    if framing == FRAMING_CONTENT_LENGTH:
        return f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload
    return payload + b"\n"


def _is_parse_error(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and "method" not in message
        and isinstance(message.get("error"), dict)
        and message["error"].get("code") == -32700
    )


def serve(
    handler: Any,
    stdin: BinaryIO,
    stdout: BinaryIO,
    framing: str = FRAMING_NEWLINE,
) -> None:
    reader = read_content_length_message if framing == FRAMING_CONTENT_LENGTH else read_newline_message
    while True:
        message = reader(stdin)
        if message is None:
            sys.stderr.write("[INFO] EOF received, shutting down.\n")
            return
        response = message if _is_parse_error(message) else handler.handle_message(message)
        if response is not None:
            stdout.write(encode_message(response, framing))
            stdout.flush()


def _resolve_framing(legacy_flag: bool) -> str:
    if legacy_flag:
        return FRAMING_CONTENT_LENGTH
    env_value = str(os.getenv("MCP_STDIO_FRAMING", "")).strip().lower()
    if env_value in {FRAMING_CONTENT_LENGTH, "content_length", "lsp"}:
        return FRAMING_CONTENT_LENGTH
    return FRAMING_NEWLINE


def main(argv: Optional[list[str]] = None) -> None:
    """Main stdio loop."""
    parser = argparse.ArgumentParser(description="llm-context MCP server (stdio)")
    parser.add_argument(
        "--legacy-content-length",
        action="store_true",
        help="Use LSP-style Content-Length framing instead of newline-delimited JSON",
    )
    args = parser.parse_args(argv)
    framing = _resolve_framing(args.legacy_content_length)

    _redirect_stderr_to_log()
    handler = MCPHandler(config_path=_resolve_runtime_config_path())
    sys.stderr.write(f"[INFO] MCP Server (stdio, framing={framing}) started. Waiting for messages...\n")
    serve(handler, sys.stdin.buffer, sys.stdout.buffer, framing)


if __name__ == "__main__":
    main()
