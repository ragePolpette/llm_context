# llm-context: Registry Drift Risk and Future Index Manifest

## Context

The current multi-project evolution introduced a file-based project registry and a separate runtime state file.

Current shape:

- `projects.yaml`
  - administrative source of truth for registered projects
  - project metadata
  - root path
  - indexing/retrieval profile
  - ingest/write enablement
- `projects.state.json`
  - last ingest status
  - timestamps
  - lightweight operational state

This is a good first step, but it creates a known architectural risk: the registry can drift away from the real index state.

## Drift Problem

Current chain:

`project registry -> project root path -> index state elsewhere`

The registry can say:

- project exists
- project is enabled
- last ingest succeeded

while the actual index may be:

- missing
- partial
- stale
- broken
- incompatible with current config/schema

### Example

- `projects.yaml`
  - `projectA -> /repoA`
- `projects.state.json`
  - `last_ingest_status = success`
- actual backing store
  - index rows missing or corrupted
  - chunk embeddings incomplete
  - schema/profile mismatch

In that scenario the server still believes the project is available and healthy, but retrieval may fail or silently degrade.

## Why This Happens

The current registry is an administrative catalog, not a strong contract for index integrity.

Today the system lacks a per-project artifact that says:

- what exact index was produced
- against which config/profile
- with which source fingerprint
- against which storage target/schema
- whether the read-plane can trust it

## Recommended Future Fix: Per-Project Index Manifest

Introduce an `index manifest` per project.

This should become the contract between:

- write-plane / ingest CLI
- read-plane / MCP retrieval server
- dashboard / operational visibility

## Recommended Responsibilities of the Index Manifest

For each project, track at least:

- `project_id`
- `index_version`
- `schema_version`
- `last_ingest_started_at`
- `last_ingest_completed_at`
- `last_ingest_status`
- `indexed_documents`
- `indexed_chunks`
- `indexed_symbols`
- `index_fingerprint`
- `config_fingerprint`
- `source_fingerprint` or source revision snapshot
- `store_target`
- `last_error` if ingest failed

Optional but useful:

- embedder/model identity
- chunking profile identity
- storage health check result
- stale/reindex-required flag

## Placement

Recommended first implementation:

- save one manifest per project inside the project-local operational area
- also expose a summarized copy in central state if useful for dashboard aggregation

Examples:

- per-project manifest near project runtime metadata
- central aggregated status for dashboard listing

The important architectural point is:

- registry remains the administrative catalog
- manifest becomes the index integrity record

## How the Read Plane Should Use It

The retrieval server should not trust registry presence alone.

Instead it should consider a project effectively index-ready only if:

- the project exists in the registry
- a manifest exists
- manifest status is valid
- manifest fingerprints/schema are compatible with current runtime expectations

If not:

- mark the project as stale or unavailable
- expose that clearly in health/status
- avoid pretending the project is fully ready

## Minimal Incremental Rollout

Recommended future implementation order:

1. Add manifest generation in the ingest CLI
   - no retrieval behavior changes yet
2. Surface manifest data in health/dashboard status
3. Add lightweight validation in the read-plane
4. Later, treat manifest mismatch as `stale` or `reindex required`

This keeps the change incremental and compatible with the current architecture.

## Conclusion

The current file-based project registry is acceptable as a first multi-project foundation, but it is not enough to guarantee index correctness.

The next architectural step for `llm-context` should be a per-project index manifest, so the system can distinguish:

- registered project
- ingest attempted
- ingest succeeded
- index actually trustworthy for retrieval

Without this, registry drift is a real long-term operational risk.
