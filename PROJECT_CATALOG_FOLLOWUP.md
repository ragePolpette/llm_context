# llm-context: Future Shared Project Catalog Risk

## Context

`llm-context` now has an explicit project concept, including:

- `project_id`
- `display_name`
- `root_path`
- indexing/retrieval profile data
- ingest/write operational state

This is correct for the current phase of the architecture.

## Current Risk

The same high-level domain concept, "project", now also exists in another repository:

- `llm-memory`

But the two project models are currently defined independently.

Today this is acceptable because:

- each MCP still owns its own local responsibility
- there is no central control-plane enforcing a shared project identity model

## Why This Becomes a Problem

This duplication becomes a real architectural problem when a third actor appears, for example:

- dashboard / admin plane
- ingest scheduler
- orchestration layer
- deployment/runtime controller
- another MCP that needs project-aware behavior

At that point, there is a risk of having:

- the same project represented twice with different metadata
- inconsistent naming or lifecycle state
- drift between context projects and memory projects
- duplicate creation logic
- no single place that defines whether a project "exists"

## Architectural Distinction

There are actually two different levels of meaning:

### Shared platform concept

Project identity as a platform entity:

- `workspace_id`
- `project_id`
- `display_name`
- `description`
- `status`
- `created_at`
- `updated_at`
- optional shared metadata

### Local MCP-specific concept

Project-specific metadata owned by one MCP:

- in `llm-context`: root path, ingest state, index state, retrieval profile
- in `llm-memory`: scope behavior, memory counts, memory-specific metadata

The future mistake to avoid is letting each repository define both levels independently.

## Recommended Future Direction

Introduce a shared project catalog as the canonical identity layer.

The project catalog should define only shared identity and lifecycle.

Then each MCP should attach its own local metadata:

- `llm-context` keeps corpus/index metadata
- `llm-memory` keeps scope/memory metadata

This preserves separation of concerns:

- one shared concept of project identity
- multiple service-specific project extensions

## Minimal Future Shared Model

Recommended canonical project catalog fields:

- `workspace_id`
- `project_id`
- `display_name`
- `description`
- `status`
- `metadata`
- `created_at`
- `updated_at`

Everything else should remain service-specific unless there is a clear platform reason to centralize it.

## Why Add This Note Now

This is not yet a blocking issue.

However, it is the next architectural problem that will surface once:

- dashboard starts aggregating both systems more deeply
- project creation becomes operational
- more MCP services adopt project-aware behavior

Documenting it now helps avoid introducing a third incompatible project model later.

## Conclusion

For now:

- local project models are acceptable
- each repository can continue evolving independently

But the next cross-repo architectural step should be:

- a shared project catalog
- plus local per-service project metadata layered on top

Without that, multi-project support risks fragmenting into multiple incompatible definitions of the same concept.
