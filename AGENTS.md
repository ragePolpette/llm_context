# AGENTS.md (llm_context)

Regole locali per il modulo `llm_context` (RAG + MCP server).

## Scope

- Valido solo dentro `llm_context/`.
- Non modifica e non eredita il workflow principale del repository.
- Questo modulo resta scollegato dal routing sub-agent di `.codex/`.

## Ordine di lettura locale

1. `README.md`
2. `AGENT_GUIDE.md`
3. `config.yaml`
4. `rag_indexer/*.py` (solo file rilevanti al task)

## Obiettivo operativo

- Indicizzare codice/documenti in Postgres + pgvector.
- Esporre retrieval via MCP (`rag_context`, `rag_search`).
- Mantenere retrieval stabile, ripetibile e con auto-scope coerente.

## Regole operative

- Prima di cambiare logica retrieval, verificare impatto su:
  - `config.yaml` (`scope_map`, `default_doc_type`, pesi vector/keyword)
  - `rag_indexer/mcp_handler.py`
  - `rag_indexer/agent_context.py`
- Prima di cambiare schema/ingest, verificare `rag_indexer/db.py`, `store.py`, `ingest.py`.
- Evitare modifiche inutili a file runtime/log (`.mcp_server.lock`, `mcp_server.err.log`).

## Verifica minima

- Se tocchi retrieval/chunking: eseguire `tests/test_retrieval_v2.py` e/o `tests/test_chunking.py`.
- Se tocchi MCP: verificare avvio server e risposta tool (`tools/list`, `tools/call`).

## Nota isolamento

- Nessuna dipendenza dal workflow "Codex Sub-Agent Playbook" root.
- Decisioni locali di `llm_context` non cambiano regole globali del monorepo.
