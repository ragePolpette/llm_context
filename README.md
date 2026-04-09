# llm_context

`llm_context` is a local-first MCP retrieval server for technical context from code and documents.

It is built for RAG-style lookup over repositories and technical document sets, with explicit project boundaries, incremental ingest, and retrieval surfaces aimed at agent workflows. It is not a persistent operational memory system; that role belongs to `llm-memory`.

## What It Does

- indexes code and documents into a local vector-backed retrieval store
- supports technical retrieval through MCP tools such as `rag_context`, `rag_search`, and `symbol_search`
- separates read-plane retrieval from write-plane ingest
- supports explicit project discovery and multi-project configuration
- keeps retrieval focused on local infrastructure and local embeddings

## Why It Exists

Agents often need precise technical context without loading an entire repository into a prompt.

`llm_context` is meant to provide:

- local retrieval over source code and technical docs
- explicit project scoping instead of implicit workspace leakage
- incremental ingest instead of full reindex every time
- a retrieval-oriented MCP surface that stays separate from operational memory

## Core Concepts

- Read-plane vs write-plane: retrieval is exposed through MCP; ingest remains an operational capability
- Single-project vs multi-project mode: legacy-safe defaults or explicit project selection
- Incremental ingest: only changed files need to be reprocessed
- Local embeddings: the intended path is local embedding generation, not hosted inference

## MCP Surface

Main retrieval tools:

- `rag_context`
- `rag_search`
- `symbol_search`
- `list_projects`
- `get_project_info`
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

## Local Run

Requirements:

- Python 3.10+
- PostgreSQL with `pgvector`
- local embedding runtime support

Install:

```bash
pip install -e .
```

Run the MCP server:

```bash
python mcp_server.py
```

Run the local HTTP server:

```bash
python mcp_server_http.py
```

Example operational CLI flows:

```bash
python cli.py --config config.yaml list-projects --json
python cli.py --config config.yaml ingest --dsn <dsn> --project-id <project_id>
python cli.py --config config.yaml ingest-enabled-projects --dsn <dsn>
```

## Configuration Model

Key files:

- `config.example.yaml`
- `config.local.example.env`
- `projects.example.yaml`

Important runtime ideas:

- `multi_project_enabled`
- `default_project_id`
- `projects_registry_path`
- `projects_state_path`
- `ingest_enabled`
- `write_enabled`

## Project Status

This repository is in active development. The current runtime already supports real local retrieval workflows, but the repo is still evolving toward a cleaner public-facing shape and stronger multi-project ergonomics.

## Related Repositories

- `llm-memory`: persistent operational memory layer
- `mcp-dashboard`: local control plane and observability surface

## Development Process

Built with AI-assisted workflows, while architecture, tradeoffs, integration, review, and validation were directed by the author.
