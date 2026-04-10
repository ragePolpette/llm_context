# Agent Guide - Querying the RAG Memory DB

Questo documento spiega come un agente LLM deve interrogare Postgres per ottenere contesto rilevante.
Il DB contiene i chunk indicizzati dal progetto (schema v2).

## Prerequisiti

- PostgreSQL + pgvector raggiungibile dal DSN
- Indice gia' creato e popolato con `ingest`
- Embedding locali attivi (default: 384 dimensioni)

Nota operativa:

- il servizio MCP gira come processo Python normale
- Docker non e' richiesto per il servizio
- se il database e' in Docker e' una scelta di provisioning del DB, non del MCP
- l'agente non deve assumere `localhost:5432` o un container specifico; deve usare il DSN runtime fornito dal launcher o dalla dashboard

## Surface MCP consigliato

Ordine d'uso raccomandato:

- `context_info`: capire ruoli, limiti, quick start e regole decisionali dei tool disponibili
- `rag_context`: ottenere il package principale per iniziare a lavorare
- `symbol_search`: disambiguare simboli, signature e linee esatte
- `rag_search`: approfondire risultati raw o fare ricerca mirata, usando i gruppi per file e gli hint di investigazione restituiti dal tool

## Opzione A (consigliata): helper Python

Usa l'helper `rag_indexer/agent_context.py` che incapsula retrieval + formattazione contesto.

```python
from rag_indexer.agent_context import build_context
from rag_indexer.embedder import LocalSentenceTransformerEmbedder

DSN = "postgresql://<user>:<password>@<host>:5432/<database>"
PROJECT_ID = "<project_id>"
embedder = LocalSentenceTransformerEmbedder(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

context, results = build_context(
    dsn=DSN,
    embedder=embedder,
    embedding_dim=384,
    project_id=PROJECT_ID,
    query_text="come funziona il modulo Fiscobot?",
    top_k=8,
)

print(context)
```

Poi usa `context` come input nel prompt dell'agente:

```
System: Sei un assistente tecnico.
User: Domanda utente...
Context:
<context generato>
```

Nota: `embedding_dim` deve coincidere con la dimensione usata dal DB (384).

## Opzione A2: CLI "context"

Se l'agente esegue comandi di sistema, puo' usare il comando `context` per ottenere
direttamente il testo da inserire nel prompt:

```bash
python cli.py context \
  --dsn "postgresql://<user>:<password>@<host>:5432/<database>" \
  --project-id <project_id> \
  --text "come funziona bpofh?" \
  --top-k 8 \
  --path-prefix "librerie\\BpoFH\\" \
  --embedder local-st
```

Auto-scope: se il testo contiene parole chiave note (es. "bpofh"),
il comando applica automaticamente il filtro sul path. Se il testo contiene un path file,
il tool lo rileva e filtra automaticamente.

Oppure usa direttamente un file/dir:

```bash
python cli.py context \
  --dsn "postgresql://<user>:<password>@<host>:5432/<database>" \
  --project-id <project_id> \
  --text "spiegami il flusso" \
  --top-k 8 \
  --file "librerie\\BpoFH\\Attivita.cs" \
  --embedder local-st
```

Il comando stampa solo il contesto. Se vuoi anche i risultati raw:

```bash
python cli.py context --print-results ...
```

## Opzione B: query diretta al DB (psycopg, schema v2)

Se vuoi gestire tu l'embedding e la query:

```python
import psycopg
from rag_indexer.embedder import LocalSentenceTransformerEmbedder

DSN = "postgresql://<user>:<password>@<host>:5432/<database>"
PROJECT_ID = "<project_id>"
embedder = LocalSentenceTransformerEmbedder(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

query_text = "come funziona X?"
embedding = embedder.embed(query_text)

sql = """
SELECT c.chunk_id, d.path, c.chunk_index, c.text,
       (e.embedding <=> %s) AS distance
FROM chunk_embeddings e
JOIN chunks c ON c.chunk_id = e.chunk_id
JOIN documents d ON d.doc_id = c.doc_id
WHERE d.repo_id = %s
  AND d.deleted_at IS NULL
ORDER BY e.embedding <=> %s
LIMIT %s
"""

with psycopg.connect(DSN) as conn:
    with conn.cursor() as cur:
        cur.execute(sql, [embedding, PROJECT_ID, embedding, 8])
        rows = cur.fetchall()

for row in rows:
    print(row[1], row[2], row[5])
```

## Linee guida per l'agente

- Usa sempre `project_id` esplicito e gia' registrato nel registry per isolare il contesto.
- Usa `doc_type` per filtrare (es. `code`) se vuoi ridurre rumore.
- Non usare dati di test se non richiesti (limita `include_dirs` nel config se necessario).

## Nota sulla privacy

Con `local-st` gli embedding sono calcolati in locale.
Non c'e' invio dati all'esterno (a parte il download iniziale del modello).

