# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Security

- MCP tools no longer let clients choose the embedder, the sentence-transformers model or the embedding dimension. These are server-side settings (`LLM_CONTEXT_EMBEDDER`, `LLM_CONTEXT_LOCAL_MODEL`, `LLM_CONTEXT_EMBEDDING_DIM`). A client value equal to the server setting is ignored with a warning; any other value is rejected. The arguments were removed from the tool schemas. Previously a client could make the server download and load an arbitrary model.
- `top_k`, symbol `limit`, `max_chars` and the `query_text` length are capped (`LLM_CONTEXT_MAX_TOP_K`, `LLM_CONTEXT_MAX_SYMBOL_LIMIT`, `LLM_CONTEXT_MAX_CHARS_LIMIT`, `LLM_CONTEXT_MAX_QUERY_CHARS`). Non-integer values are rejected.
- Optional bearer authentication for the HTTP server through `MCP_AUTH_TOKEN`. When set, every endpoint, including `/health`, requires `Authorization: Bearer <token>`. The server refuses to bind to a non-loopback `MCP_HOST` without a token.
- The HTTP server no longer kills other processes. The netstat, wmic, lsof and ps based "kill the previous instance" logic was removed: if the lock is held or the port is busy the server exits with an error. `scripts/stop_http_server.bat` only stops the PID recorded in `.mcp_server.pid`.
- `/rpc` no longer prints full request and response bodies to stdout; it logs method, id and size at debug level.

### Changed

- The stdio transport now uses newline-delimited JSON as required by the MCP specification. The previous `Content-Length` framing is available with `--legacy-content-length` or `MCP_STDIO_FRAMING=content-length`. Responses are serialized with the UUID-aware encoder.
- HTTP tool calls and `/health` run in a worker thread (`asyncio.to_thread`) instead of blocking the event loop.
- Full ingest no longer deletes the whole project index before re-indexing. Each document is replaced in a single transaction and documents missing from the source tree are deleted at the end of the run, so readers never see an empty index.
- Indexed paths, path prefixes and scanner paths use forward slashes. Path-prefix filters also match rows indexed with backslashes; a full re-ingest rewrites old rows.
- Keyword search uses the configurable `fts_config` (default `simple`) instead of the hard-coded `italian_unaccent`. Existing Italian deployments should set `fts_config: italian_unaccent`; `init-db-v2` only creates that configuration when it is requested.
- Domain-specific mappings moved to configuration with empty defaults: `scope_map`, `path_aliases`, `product_targets` and `product_aliases` can be set globally or per project in the registry. `assets_root_dir` and `assets_template_dir` now default to empty. Without `product_targets`, `map_work_item_to_codebase` reports `product_target: unknown`.
- The project registry is reloaded when the registry, state or manifest files change on disk, without restarting the server.
- The query cache key now includes `file`, `max_chars`, `auto_scope` and the server embedder identity, and is computed from validated values.
- Config values are coerced to their declared types (for example `ingest_enabled: "false"` is now `false` instead of a truthy string).
- `config.yaml` now contains neutral defaults and points to `projects.yaml` with state under `.local/runtime/`. `config.example.yaml` is a generic example. `projects.yaml` and `projects.state.json` are git-ignored.
- `ingest` and `query` default to the `local-st` embedder instead of `dummy`.
- Package renamed to `llm-context`; the version (`0.2.0`) is defined once in `rag_indexer.__version__` and reported in `serverInfo`. A `llm-context` console script provides `serve`, `ingest`, `query` and the other commands. The CLI moved to `rag_indexer/cli.py`; `cli.py` remains as a thin wrapper.
- `aiohttp` is now a declared dependency.
- Tool descriptions, tool output, readiness messages and errors are in English.
- `reingest_completo.bat` was replaced by `reingest_full.bat`, which uses `config.ingest.yaml`, defaults the root to the repository folder and never prints the DSN.
- `scripts/` was reduced to the wrappers in use, in English. The `*_rework.bat` aliases were removed and `smoke_test_rework_http.py` was renamed to `smoke_test_http.py`.
- `mcp_proxy.py` reads the target URL from `LLM_CONTEXT_PROXY_URL` and sends `MCP_AUTH_TOKEN` when set.

### Removed

- `GeminiEmbedder` stub and the `--gemini-api-key` and `--gemini-model` CLI flags.
- The unused `rerank_enabled` setting (existing config files that set it still load).

### Development

- Ruff is configured in `pyproject.toml` and runs in CI.
