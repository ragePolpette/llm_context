# llm_context

[![CI](https://github.com/ragePolpette/llm_context/actions/workflows/ci.yml/badge.svg)](https://github.com/ragePolpette/llm_context/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)

`llm_context` is a local-first MCP retrieval server for technical context from code and documents.

It is built for RAG-style lookup over repositories and technical document sets, with explicit project boundaries, incremental ingest, and retrieval surfaces aimed at agent workflows. It is not a persistent operational memory system; that role belongs to `llm-memory`.

## Overview

- indexes code and documents into a local vector-backed retrieval store
- supports technical retrieval through MCP tools such as `rag_context`, `rag_search`, and `symbol_search`
- separates read-plane retrieval from write-plane ingest
- supports project discovery through a registry and explicit project selection
- keeps retrieval focused on local infrastructure and local embeddings

## Why It Exists

Agents often need precise technical context without loading an entire repository into a prompt.

`llm_context` is meant to provide:

- local retrieval over source code and technical docs
- explicit project scoping instead of implicit workspace leakage
- incremental ingest instead of full reindex every time
- a retrieval-oriented MCP surface that stays separate from operational memory

## Design Tradeoffs

This project started from a real codebase where a single-project SQLite setup was still barely workable, but sustained indexing and retrieval were already close to the point where an embedded store stopped being comfortable.

The main architectural tradeoff was to prioritize ingestion throughput, retrieval performance, concurrency, and multi-project growth over the simplicity and portability of SQLite.

That led to PostgreSQL with `pgvector`:

- metadata and vector storage stay in one operational stack
- multi-project indexing has a more credible persistence model
- retrieval can scale beyond the limits of a single embedded local file
- the storage layer is better aligned with sustained ingest workloads

The cost is deliberate:

- the project is less portable than an all-SQLite setup
- local setup is heavier
- the runtime depends on a database that is worth operating only if the retrieval surface is large enough to justify it

In other words, this repository trades some simplicity for a storage model that better matches real indexing pressure and multi-project retrieval.

## Core Concepts

- Read-plane vs write-plane: retrieval is exposed through MCP; ingest remains an operational capability
- Explicit project registry: every ingest and retrieval operation is project-scoped
- Incremental ingest: only changed files need to be reprocessed
- Local embeddings: the intended path is local embedding generation, not hosted inference

## MCP Surface

Retrieval tools:

- `rag_context`
- `rag_search`
- `rag_full_context`
- `symbol_search`
- `map_work_item_to_codebase`

Project and runtime tools:

- `list_projects`
- `get_project_info`
- `get_project_snapshot`
- `context_info`

The ingest path is intentionally not exposed as a standard always-on MCP tool surface.

## Architecture

Main runtime areas:

- `rag_indexer/`: scanning, chunking, embedding, storage, and retrieval logic
- `mcp_server.py`: MCP stdio transport
- `mcp_server_http.py`: local HTTP transport
- `rag_indexer/cli.py` (`llm-context`, also `cli.py`): server launcher and operational ingest and maintenance CLI
- `config.example.yaml`: example configuration for local setup
- `projects.example.yaml`: project registry example for multi-project mode

## Data Flow

```text
files and documents
   |
   v
scanner -> chunking -> embeddings -> store
   |
   v
retrieval + filtering + project scope
   |
   v
MCP context tools
```

## Quick Start

Requirements:

- Python 3.10 or newer
- PostgreSQL with the `pgvector` extension
- local embedding runtime support (`sentence-transformers`, installed as a dependency)

Install:

```bash
pip install -e .
```

Optional symbol extraction (tree-sitter):

```bash
pip install -e ".[symbols]"
```

The package installs an `llm-context` command; `python cli.py ...` is equivalent.

Create the schema, then ingest a project:

```bash
cp projects.example.yaml projects.yaml
llm-context --config config.yaml init-db-v2 --dsn <dsn>
LLM_CONTEXT_INGEST_ENABLED=true llm-context --config config.yaml ingest --dsn <dsn> --project-id <project_id>
```

`ingest` and `query` default to the `local-st` embedder.

Run the MCP server (stdio):

```bash
llm-context serve            # or: python mcp_server.py
```

The stdio transport follows the MCP specification: one JSON-RPC message per line (newline-delimited JSON). Clients that still use LSP-style `Content-Length` framing can opt in with `llm-context serve --legacy-content-length` (or `python mcp_server.py --legacy-content-length`, or `MCP_STDIO_FRAMING=content-length`).

Run the local HTTP server (binds to `127.0.0.1:8765` by default; `/mcp`, `/rpc` and `/health`):

```bash
llm-context serve --transport http   # or: python mcp_server_http.py
```

Tool calls run in a worker thread, so a slow query does not block other requests. Only one instance per checkout can run (file lock); a running instance is never killed automatically.

Other CLI commands: `list-projects`, `query`, `context`, `export-snapshot`. Use `llm-context <command> --help` for options.

The `scripts/` directory contains a few Windows `.bat` wrappers (`init_db`, `ingest`, `start_http_server`, `start_stdio_server`, `status`, `stop_http_server`) and `smoke_test_http.py`. `reingest_full.bat` runs a full re-ingest of this repository with `config.ingest.yaml`.

## Configuration

Key files:

- `config.yaml`: tracked, neutral runtime defaults. Do not put machine-specific values here.
- `config.example.yaml`: a commented example for a fictional repository. Copy it to `config.local.yaml` (untracked, starts with `extends: config.yaml`) and point `LLM_CONTEXT_CONFIG_PATH` or `--config` at it.
- `projects.example.yaml`: project registry template. Copy it to `projects.yaml`, which is git-ignored because it describes local paths.
- `.env.example`: environment variables (DSN, embedder, limits, transport security, model cache)

Runtime state (`projects.state.json` and `project_manifests/`) is written under `.local/runtime/` by default and is git-ignored. `config.rework.yaml` and `config.ingest.yaml` describe the self-indexing setup of this repository (`projects.rework.yaml`).

Important settings:

- `projects_registry_path`
- `projects_state_path`
- `ingest_enabled` (default `false`)
- `write_enabled` (default `false`)
- `fts_config` (default `simple`)
- explicit `project_id` on every ingest and MCP retrieval call

Configuration files can extend one another with `extends:`. Values are coerced to the declared types, so `ingest_enabled: "false"` is a boolean `false`. Environment variables such as `LLM_CONTEXT_CONFIG_PATH`, `LLM_CONTEXT_INGEST_ENABLED` and `LLM_CONTEXT_WRITE_ENABLED` override file values.

### Embedder

The embedder (`LLM_CONTEXT_EMBEDDER`), model (`LLM_CONTEXT_LOCAL_MODEL`) and dimension (`LLM_CONTEXT_EMBEDDING_DIM`, default `embedding_dim`) are server-side settings. MCP tools no longer accept `embedder`, `local_model` or `embedding_dim` from clients: a value equal to the server setting is ignored, any other value is rejected.

Client-controlled sizes are capped: `top_k` (`LLM_CONTEXT_MAX_TOP_K`, default 50), symbol `limit` (`LLM_CONTEXT_MAX_SYMBOL_LIMIT`, default 200), `max_chars` (`LLM_CONTEXT_MAX_CHARS_LIMIT`, default 100000) and `query_text` length (`LLM_CONTEXT_MAX_QUERY_CHARS`, default 8000).

### Project-specific mappings

Domain-specific behavior lives in configuration, with empty defaults. Each mapping can be set globally in the config file or per project in the registry; a non-empty project value wins.

- `scope_map`: query keywords that narrow retrieval to a path prefix
- `path_aliases`: path fragments rewritten in queries (`{from: to}`)
- `product_targets` and `product_aliases`: used by `map_work_item_to_codebase` to infer the product touched by a work item. Without them `product_target` is `unknown`.

```yaml
projects:
  - project_id: shop
    root_path: ../acme-shop
    scope_map:
      src/billing/: [billing, invoice]
    product_targets:
      storefront: [storefront, web]
```

The project registry is reloaded automatically when `projects.yaml`, the state file or a manifest changes on disk (checked on `/health`, `list_projects`, `get_project_info`, `context_info` and on every project-scoped tool call).

### Paths

Indexed paths and path prefixes use forward slashes (`src/api/Controller.cs`) on every platform. Existing indexes with backslash paths keep working for path-prefix filters; a full re-ingest rewrites them in posix form.

### Keyword search language

Keyword search uses the PostgreSQL text search configuration named by `fts_config` (default `simple`, which works for any language; `english` and the other built-in configurations are also valid). The value is used both when chunks are written and when they are queried, so re-ingest after changing it.

For Italian with accent folding, set `fts_config: italian_unaccent`. `init-db-v2` creates it when that value is configured; to create it manually:

```sql
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE TEXT SEARCH CONFIGURATION italian_unaccent (COPY = italian);
ALTER TEXT SEARCH CONFIGURATION italian_unaccent
  ALTER MAPPING FOR hword, hword_part, word WITH unaccent, italian_stem;
```

### Ingest

A full ingest (without `--incremental`) no longer deletes the project index up front. Every file is re-indexed and each document is replaced in a single transaction, so queries keep seeing the previous version until the new one is committed. Documents that disappeared from the source tree are removed at the end of the run.

## Security

The HTTP server binds to localhost by default and checks `Host` and `Origin` headers against allowlists. Set `MCP_AUTH_TOKEN` to require `Authorization: Bearer <token>` on every HTTP endpoint; the server refuses to bind to a non-loopback `MCP_HOST` without it. `mcp_proxy.py` and `scripts/smoke_test_http.py` send the token when `MCP_AUTH_TOKEN` is set. See [SECURITY.md](SECURITY.md) for the threat model and reporting instructions.

## Development

```bash
pip install -e ".[dev]"
ruff check .
python -m pytest -q
```

Tests do not require a running database; tests that need one are skipped. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Project Status

This repository is in active development. The current runtime already supports real local retrieval workflows, but the repo is still evolving toward a cleaner public-facing shape and stronger multi-project ergonomics.

Built with AI-assisted workflows, while architecture, tradeoffs, integration, review, and validation were directed by the author.

## Related Projects

- `llm-memory`: persistent operational memory layer
- `mcp-dashboard`: local control plane and observability surface

## License

MIT. See [LICENSE](LICENSE).
