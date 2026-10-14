# Security Policy

## Scope

This policy covers the code in this repository: the MCP servers (`mcp_server.py`, `mcp_server_http.py`), the `rag_indexer` package and the `cli.py` ingest and maintenance CLI.

## Threat model

`llm_context` is designed as a local component that runs on a developer machine next to a PostgreSQL instance with `pgvector`.

- The HTTP server binds to `127.0.0.1` by default and validates the `Host` and `Origin` headers against allowlists (`MCP_ALLOWED_HOSTS`, `MCP_ALLOWED_ORIGINS`). The server has no authentication layer of its own.
- Exposing the HTTP server beyond localhost, for example by changing `MCP_HOST`, is outside the intended deployment. If you do it, you are responsible for adding network-level access control.
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
