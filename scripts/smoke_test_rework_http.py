from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _http_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _rpc_call(base_url: str, *, request_id: int, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments or {},
        },
    }
    return _http_json(f"{base_url}/rpc", method="POST", payload=payload)


def _wait_for_health(base_url: str, *, timeout_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _http_json(f"{base_url}/health")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Server health check timed out: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test del runtime HTTP canonico di llm-context")
    parser.add_argument("--config-path", default="config.rework.yaml")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--runtime-name", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--exercise-retrieval", action="store_true")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    config_path = (root_dir / args.config_path).resolve()
    env = os.environ.copy()
    env.setdefault("LLM_CONTEXT_CONFIG_PATH", str(config_path))
    if args.dsn:
        env["LLM_CONTEXT_DSN"] = args.dsn
    if args.runtime_name:
        env["LLM_CONTEXT_RUNTIME_NAME"] = args.runtime_name
    env.setdefault("LLM_CONTEXT_RUNTIME_NAME", "llm-context")
    env.setdefault("PYTHONPATH", str(root_dir))
    env["LLM_CONTEXT_EMBEDDER"] = "local-hash"
    if args.host:
        env["MCP_HOST"] = args.host
    if args.port is not None:
        env["MCP_PORT"] = str(args.port)
    if not env.get("LLM_CONTEXT_DSN"):
        raise RuntimeError(
            "LLM_CONTEXT_DSN non impostato. Passalo al processo oppure usa --dsn nello smoke test."
        )

    host = env.get("MCP_HOST", "127.0.0.1")
    port = int(env.get("MCP_PORT", "8766"))
    base_url = f"http://{host}:{port}"

    process = subprocess.Popen(
        [sys.executable, "-u", "mcp_server_http.py"],
        cwd=root_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        health = _wait_for_health(base_url, timeout_sec=args.timeout_sec)
        if health.get("runtime_name") != env["LLM_CONTEXT_RUNTIME_NAME"]:
            raise RuntimeError(f"Unexpected runtime_name in health payload: {health}")
        storage_target = health.get("storage_target") or {}
        if not storage_target.get("database"):
            raise RuntimeError(f"Health payload does not expose storage_target.database: {health}")
        if storage_target.get("dedicated_candidate") is False:
            raise RuntimeError(f"Storage target does not look dedicated: {storage_target}")

        tools_list = _http_json(
            f"{base_url}/rpc",
            method="POST",
            payload={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        tools = tools_list.get("result", {}).get("tools", [])
        tool_names = {item.get("name") for item in tools if isinstance(item, dict)}
        required_tools = {
            "rag_context",
            "rag_search",
            "symbol_search",
            "map_work_item_to_codebase",
            "context_info",
            "list_projects",
            "get_project_info",
        }
        missing_tools = sorted(required_tools - tool_names)
        if missing_tools:
            raise RuntimeError(f"Missing tools in tools/list: {missing_tools}")

        context_info = _rpc_call(base_url, request_id=2, name="context_info")
        context_payload = context_info.get("result", {}).get("content", [])
        if not context_payload:
            raise RuntimeError("context_info returned no content")

        list_projects = _rpc_call(base_url, request_id=3, name="list_projects")
        projects_payload = json.loads(list_projects["result"]["content"][0]["text"])
        if projects_payload.get("count", 0) < 1:
            raise RuntimeError("list_projects returned no registered projects")
        project_id = projects_payload["projects"][0]["project_id"]

        project_info = _rpc_call(
            base_url,
            request_id=4,
            name="get_project_info",
            arguments={"project_id": project_id},
        )
        project_info_payload = json.loads(project_info["result"]["content"][0]["text"])
        if project_info_payload.get("project_id") != project_id:
            raise RuntimeError("get_project_info returned unexpected project_id")

        if args.exercise_retrieval:
            _rpc_call(
                base_url,
                request_id=5,
                name="rag_context",
                arguments={"project_id": project_id, "query_text": "mcp handler runtime"},
            )
            _rpc_call(
                base_url,
                request_id=6,
                name="rag_search",
                arguments={"project_id": project_id, "query_text": "runtime config"},
            )
            _rpc_call(
                base_url,
                request_id=7,
                name="symbol_search",
                arguments={"project_id": project_id, "name": "MCPHandler"},
            )

        print(
            json.dumps(
                {
                    "status": "ok",
                    "base_url": base_url,
                    "project_id": project_id,
                    "storage_target": storage_target,
                },
                indent=2,
            )
        )
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())


