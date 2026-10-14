from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

import mcp_server
from rag_indexer import __version__

ROOT = Path(__file__).resolve().parent.parent


def _write_config(tmp_path: Path) -> Path:
    projects_path = tmp_path / "projects.yaml"
    projects_path.write_text(
        yaml.safe_dump({"projects": [{"project_id": "alpha", "root_path": "repo"}]}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects_registry_path": projects_path.name,
                "projects_state_path": "projects.state.json",
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _server_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "LLM_CONTEXT_DSN": "postgresql://user:pass@127.0.0.1:1/none",
            "LLM_CONTEXT_EMBEDDER": "local-hash",
            "LLM_CONTEXT_CONFIG_PATH": str(_write_config(tmp_path)),
            "LLM_CONTEXT_STDIO_LOG_PATH": str(tmp_path / "stdio.log"),
            "MCP_MODELS_DIR": str(tmp_path / "models"),
            "PYTHONPATH": str(ROOT),
        }
    )
    env.pop("MCP_STDIO_FRAMING", None)
    return env


def _requests() -> list[dict]:
    return [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t"}},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]


def test_stdio_server_speaks_newline_delimited_json(tmp_path):
    stdin = "".join(json.dumps(message) + "\n" for message in _requests())
    completed = subprocess.run(
        [sys.executable, str(ROOT / "mcp_server.py")],
        input=stdin.encode("utf-8"),
        capture_output=True,
        env=_server_env(tmp_path),
        timeout=60,
        check=True,
    )

    lines = completed.stdout.decode("utf-8").splitlines()
    assert len(lines) == 2
    responses = [json.loads(line) for line in lines]
    assert responses[0]["id"] == 1
    assert responses[0]["result"]["serverInfo"] == {"name": "llm-context-mcp", "version": __version__}
    assert responses[1]["id"] == 2
    tool_names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert {"rag_context", "symbol_search", "list_projects"} <= tool_names


def test_stdio_server_supports_legacy_content_length_flag(tmp_path):
    payload = b""
    for message in _requests():
        body = json.dumps(message).encode("utf-8")
        payload += f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    completed = subprocess.run(
        [sys.executable, str(ROOT / "mcp_server.py"), "--legacy-content-length"],
        input=payload,
        capture_output=True,
        env=_server_env(tmp_path),
        timeout=60,
        check=True,
    )

    assert completed.stdout.startswith(b"Content-Length: ")
    reader = io.BytesIO(completed.stdout)
    first = mcp_server.read_content_length_message(reader)
    second = mcp_server.read_content_length_message(reader)
    assert first["id"] == 1
    assert second["id"] == 2


def test_newline_reader_skips_blank_lines_and_reports_parse_errors():
    stream = io.BytesIO(b"\n{not json}\n\n{\"jsonrpc\": \"2.0\", \"id\": 3, \"method\": \"ping\"}\n")

    first = mcp_server.read_newline_message(stream)
    second = mcp_server.read_newline_message(stream)
    third = mcp_server.read_newline_message(stream)

    assert first["error"]["code"] == -32700
    assert second["id"] == 3
    assert third is None


def test_serve_answers_parse_errors_without_calling_handler():
    class Handler:
        def __init__(self):
            self.calls = []

        def handle_message(self, message):
            self.calls.append(message)
            return {"jsonrpc": "2.0", "id": message["id"], "result": {}}

    handler = Handler()
    stdout = io.BytesIO()
    mcp_server.serve(handler, io.BytesIO(b"garbage\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"x\"}\n"), stdout)

    lines = stdout.getvalue().decode("utf-8").splitlines()
    assert json.loads(lines[0])["error"]["code"] == -32700
    assert json.loads(lines[1])["id"] == 1
    assert len(handler.calls) == 1


def test_encode_message_serializes_uuid_on_a_single_line():
    value = uuid.uuid4()
    encoded = mcp_server.encode_message({"jsonrpc": "2.0", "id": 1, "result": {"doc": value, "text": "a\nb"}})

    assert encoded.endswith(b"\n")
    assert encoded.count(b"\n") == 1
    assert json.loads(encoded)["result"]["doc"] == str(value)
