# claude.md — llm_context

## Project summary

Server MCP (Model Context Protocol) in Python per il **retrieval RAG di contesto tecnico** (codice + documenti) indicizzato su **Postgres + pgvector**. Dedicato esclusivamente al retrieval di contesto codice/documentazione (non è un sistema di memoria operativa — quello è `llm-memory`).

Funzionalità principali:
- Indicizzazione incrementale di una codebase in chunk con embedding locali (SentenceTransformers, no invio dati)
- Ricerca ibrida: vector (cosine) + keyword (tsvector) con pesi configurabili
- Scope auto-applicato tramite `scope_map`: keyword nel testo → filtro path automatico
- Schema v2: tabelle `documents` + `chunks` + `chunk_embeddings` con indice ivfflat
- CLI per ingest, query, context e init DB
- Proxy MCP HTTP su `127.0.0.1:8765`

Attualmente indicizza la codebase di **BpoPilot** (ASP.NET + C# + JS).

## Quickstart

```bash
cd llm_context

# Prerequisiti: Docker per Postgres+pgvector
docker start pgvector
# oppure primo avvio:
# docker run --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg15

# Installa dipendenze
pip install -e .

# Init DB v2 (solo prima volta o dopo reset)
python cli.py init-db-v2 \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --embedding-dim 384

# Ingest incrementale
python cli.py --verbose ingest --incremental \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --root ./ \
  --embedder local-st

# Avvia server MCP HTTP
python -u mcp_server_http.py
# → http://127.0.0.1:8765
```

## Architecture overview

```
Client MCP / CLI
    |
mcp_server_http.py / mcp_server.py  (HTTP/stdio MCP proxy)
    |
cli.py  (comandi: init-db-v2, ingest, query, context)
    |
rag_indexer/
    |-- scanner.py      (percorre directory, applica include/exclude)
    |-- chunking.py     (chunk MD per heading, codice per lunghezza)
    |-- embedder.py     (LocalSentenceTransformerEmbedder | GeminiEmbedder)
    |-- store.py        (upsert documents/chunks/embeddings in Postgres v2)
    |-- retrieval.py    (hybrid search: vector + keyword, filtri, dedup)
    |-- agent_context.py (helper: build_context() per agenti)
    |
Postgres + pgvector
    |-- documents (repo_id, path_norm, content_hash, metadata)
    |-- chunks (offset, linee, section_path, tsvector)
    |-- chunk_embeddings (vettori + modello)
```

## Key modules / folders

| Percorso | Ruolo |
|---|---|
| `mcp_server_http.py` | Entry point HTTP MCP (porta 8765) |
| `mcp_server.py` | Entry point stdio MCP |
| `mcp_proxy.py` | Proxy MCP |
| `cli.py` | CLI: init-db-v2, ingest, query, context |
| `config.yaml` | Configurazione principale (scope, chunk, embedding, include/exclude) |
| `rag_indexer/scanner.py` | Scanner filesystem con include/exclude |
| `rag_indexer/chunking.py` | Chunker MD + codice |
| `rag_indexer/embedder.py` | Embedding provider (locale ST / Gemini opzionale) |
| `rag_indexer/store.py` | Store v2 Postgres (upsert incrementale) |
| `rag_indexer/retrieval.py` | Ricerca ibrida, filtri, dedup |
| `rag_indexer/agent_context.py` | Helper `build_context()` per agenti |
| `config.local.example.env` | Template variabili ambiente |
| `AGENT_GUIDE.md` | Guida uso per agenti |
| `tests/` | Test (Non determinabile la copertura attuale) |

## Dependencies & tooling

- **Python**: ≥3.10
- **DB**: `psycopg[binary]>=3.1`, `psycopg_pool>=3.1`, `pgvector>=0.2.3`
- **Embedding**: `sentence-transformers>=2.6` (modello default: `paraphrase-multilingual-MiniLM-L12-v2`, dim 384)
- **Config**: `python-dotenv>=1.0`, `pyyaml>=6.0`
- **Runtime infra**: Docker (Postgres + pgvector:pg15)
- **Build**: `hatchling`
- Modello locale: ~500MB RAM + ~500MB disco per pesi

## Configuration

### `config.yaml` (parametri principali)

| Chiave | Valore attuale | Note |
|---|---|---|
| `embedding_dim` | `384` | Deve corrispondere al modello |
| `chunk_size` | `1200` | Caratteri per chunk generico |
| `code_chunk_size` | `2500` | Chunk codice (.py/.js/.cs/.aspx) |
| `md_chunk_size` | `1600` | Chunk Markdown |
| `vector_weight` | `0.5` | Peso ricerca vettoriale |
| `keyword_weight` | `0.5` | Peso ricerca keyword |
| `min_score` | `0.20` | Soglia minima retrieval |
| `default_doc_type` | `code` | Filtro default retrieval |
| `include_dirs` | vedi config.yaml | Whitelist percorsi indicizzati (BpoPilot) |

### Variabili ambiente (`.env`)

| Variabile | Default | Note |
|---|---|---|
| `MCP_HOST` | `127.0.0.1` | |
| `MCP_PORT` | `8765` | |
| `MCP_SSE_ENABLED` | `false` | |
| `MCP_MODELS_DIR` | `.local/models` | Cache modelli locali |
| `HF_HOME` | `.local/models/huggingface` | Cache HuggingFace |
| `SENTENCE_TRANSFORMERS_HOME` | `.local/models/huggingface/sentence_transformers` | |

La DSN Postgres viene passata come argomento CLI (non da env).

## Common commands

```bash
# Avvio server MCP HTTP
python -u mcp_server_http.py

# Init DB v2
python cli.py init-db-v2 --dsn "postgresql://postgres:postgres@localhost:5432/postgres" --embedding-dim 384

# Ingest incrementale
python cli.py --verbose ingest --incremental \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj --root ./ --embedder local-st

# Query
python cli.py query --dsn "postgresql://..." --project-id myproj \
  --text "come funziona X?" --top-k 8 --embedder local-st

# Context per agenti
python cli.py context --dsn "postgresql://..." --project-id myproj \
  --text "bpofh" --top-k 8 --embedder local-st

# Reset completo indice (DISTRUTTIVO)
docker exec -it pgvector psql -U postgres -d postgres -c \
  "DROP TABLE IF EXISTS chunk_embeddings; DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS documents;"

# Re-ingest completo
reingest_completo.bat
```

## Operational notes

- **Porta**: 8765. Deploy: `tools/deploy-mcp-dev-to-deploy.ps1` → copia in `Binah\llm_context`.
- Il container Docker `pgvector` deve essere avviato prima del server MCP.
- L'indice ivfflat viene creato su `chunk_embeddings.embedding`: ottimale sopra ~10k vettori.
- `embedding_dim` è fisso per DB: per cambiare modello occorre reset completo + re-ingest.
- Regola operativa agenti: eseguire sempre almeno una query `context` prima di usare grep/file search.
- `assets_template_only: true` limita l'indicizzazione degli asset a `pubblico/assets/template`.
- I file `.pfx`, `.pem`, `.key`, `.cer` e `.env*` sono esclusi dall'indicizzazione per sicurezza.
- `mcp_proxy.py` è un layer aggiuntivo: verificare se usato attivamente o legacy.

## Known issues / risks

- **La DSN Postgres contiene credenziali**: non committare mai nei file di config.
- Se si cambia modello embedding, **tutto l'indice diventa incompatibile** (nessuna migrazione incrementale disponibile, solo reset + re-ingest).
- Chunking euristico (non AST): può spezzare funzioni lunghe in modo subottimale.
- `Reranker` disabilitato di default (`rerank_enabled: false`): solo hybrid + dedup.
- Il file `start_mcp_server_non serve pi ui a un caszoo.bat` nella root indica script legacy orfani.
- Log files `.err.log` / `.out.log` nella root: non tracciati, possono crescere.

## Roadmap / next actions

1. **(S)** Rimuovere/archiviare script bat legacy dalla root del progetto
2. **(M)** Aggiungere rotazione log per `.err.log` / `.out.log`
3. **(M)** Documentare procedura cambio modello embedding (reset + re-ingest) in un runbook
4. **(S)** Spostare DSN Postgres in variabile ambiente per evitare rischi di esposizione in history shell
5. **(L)** Valutare chunking AST-aware per codice C# e Python per migliorare qualità retrieval
