# Security Policy

## Scope

This policy covers the code in this repository: the MCP servers (`mcp_server.py`, `mcp_server_http.py`), the `rag_indexer` package and the `llm-context` CLI (`rag_indexer/cli.py`, also reachable through `cli.py`).

## Threat model

`llm_context` is designed as a local component that runs on a developer machine next to a PostgreSQL instance with `pgvector`.

- The HTTP server binds to `127.0.0.1` by default and validates the `Host` and `Origin` headers against allowlists (`MCP_ALLOWED_HOSTS`, `MCP_ALLOWED_ORIGINS`).
- Optional bearer authentication: when `MCP_AUTH_TOKEN` is set, every HTTP endpoint (including `/health`) requires `Authorization: Bearer <token>`. The server refuses to start on a non-loopback `MCP_HOST` unless a token is configured. The token is not a substitute for TLS: put a TLS-terminating proxy in front of the server if traffic leaves the machine.
- Exposing the HTTP server beyond localhost is outside the intended deployment. If you do it, set `MCP_AUTH_TOKEN` and add network-level access control.
- Embedder, model and embedding dimension are server-side settings only (environment and config). MCP clients cannot select a model; a client value that differs from the server setting is rejected. Client-controlled sizes (`top_k`, `limit`, `max_chars`, `query_text`, `query_embedding`, request body) are capped.
- Retrieval is project-scoped through an explicit project registry. Ingest and write operations are controlled by `ingest_enabled` and `write_enabled`, which default to `false`, and the ingest path is not exposed as a standard MCP tool.
- Indexed content is whatever you ingest. Anything indexed can be returned by retrieval tools to the connected MCP client. Use `exclude_globs` and the registry include rules to keep secrets and sensitive files out of the index.
- Database credentials are supplied through environment variables or CLI arguments, never stored in the repository. `.env` files are ignored by git; only `.env.example` is tracked.

## Reporting a vulnerability

Please do not open a public issue for security problems.

- Preferred: use GitHub Security Advisories for this repository ("Security" tab, "Report a vulnerability").
- If that is not available, open an issue that only asks for a private contact channel, without technical details.

Include the affected version or commit, reproduction steps and the expected impact. Reports are handled on a best-effort basis.

## Not covered

- Deployments that expose the server to untrusted networks without additional protection.
- Content that users choose to index, including secrets present in indexed files.
- Vulnerabilities in third-party dependencies (PostgreSQL, `pgvector`, `sentence-transformers` and others) that are not caused by how this project uses them. Report those upstream.
- Denial of service through very large local inputs on a machine you already control.
