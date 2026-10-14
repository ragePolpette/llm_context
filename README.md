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
- `mcp_server.py`: MCP transport entrypoint
- `mcp_server_http.py`: local HTTP transport
- `cli.py`: operational ingest and maintenance CLI
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

Create the schema, then ingest a project:

```bash
cp projects.example.yaml projects.yaml
python cli.py --config config.yaml init-db-v2 --dsn <dsn>
python cli.py --config config.yaml ingest --dsn <dsn> --project-id <project_id> --embedder local-st
```

Run the MCP server (stdio):

```bash
python mcp_server.py
```

Run the local HTTP server (binds to `127.0.0.1:8765` by default; `/health` and `/rpc`):

```bash
python mcp_server_http.py
```

Other CLI commands: `list-projects`, `query`, `export-snapshot`. Use `python cli.py <command> --help` for options.

The `scripts/` directory contains Windows `.bat` wrappers for the same operations, and `reingest_completo.bat` runs a full re-ingest.

## Configuration

Key files:

- `config.example.yaml`: runtime configuration template (the repository also tracks a `config.yaml` with the same defaults)
- `projects.example.yaml`: project registry for multi-project mode
- `.env.example`: environment variables (DSN, embedder, transport security, model cache)

Important settings:

- `projects_registry_path`
- `projects_state_path`
- `ingest_enabled` (default `false`)
- `write_enabled` (default `false`)
- explicit `project_id` on every ingest and MCP retrieval call

Configuration files can extend one another with `extends:`, as in `config.rework.yaml` and `config.ingest.yaml`. Environment variables such as `LLM_CONTEXT_CONFIG_PATH`, `LLM_CONTEXT_INGEST_ENABLED` and `LLM_CONTEXT_WRITE_ENABLED` override file values.

## Security

The HTTP server binds to localhost by default, checks `Host` and `Origin` headers against allowlists, and has no authentication of its own. See [SECURITY.md](SECURITY.md) for the threat model and reporting instructions.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Tests do not require a running database; tests that need one are skipped. No linter is configured. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Project Status

This repository is in active development. The current runtime already supports real local retrieval workflows, but the repo is still evolving toward a cleaner public-facing shape and stronger multi-project ergonomics.

Built with AI-assisted workflows, while architecture, tradeoffs, integration, review, and validation were directed by the author.

## Related Projects

- `llm-memory`: persistent operational memory layer
- `mcp-dashboard`: local control plane and observability surface

## License

MIT. See [LICENSE](LICENSE).
