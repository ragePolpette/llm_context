# LLM Memory Indexer (Postgres + pgvector) - Guida Operativa

Questa guida descrive come replicare l'installazione e l'indicizzazione locale che abbiamo configurato per BpoPilot.
L'obiettivo e' indicizzare solo un sottoinsieme di cartelle del progetto, usare embedding locali (nessun invio dati),
e mantenere un indice v2 con metadata ricchi e aggiornamenti incrementali.
Questo MCP serve solo per retrieval di contesto tecnico (codice/documenti): non e' un sistema di memoria operativa persistente.

## Spiegazione teorica (cosa abbiamo fatto e perche')

### Obiettivo del sistema
Costruire un indice vettoriale del progetto (codice + documenti) per abilitare Retrieval Augmented Generation (RAG).
In pratica: ogni file viene spezzato in chunk, convertito in embedding numerici, e salvato in Postgres con pgvector.
Quando fai una query, il sistema recupera i chunk piu' simili come contesto per un agente LLM.

### Glossario rapido

- **Chunk**: porzione di testo (es. 1-2k caratteri) usata come unita' di indicizzazione.
- **Embedding**: vettore numerico che rappresenta semanticamente un testo.
- **Vector DB**: database capace di fare ricerca per similarita' su embedding.
- **RAG**: tecnica che arricchisce il prompt con contesto recuperato dal DB.
- **Cosine distance**: misura di distanza tra vettori; piu' e' bassa, piu' il testo e' simile.

### Concetti chiave 

- **Perche' i chunk**: gli LLM hanno un contesto limitato. Indicizzare a blocchi consente di
  recuperare solo le parti rilevanti invece di caricare file interi.
- **Perche' gli embedding**: confrontare testo grezzo e' lento e poco robusto; gli embedding
  trasformano il testo in numeri, permettendo confronti veloci e semantici.
- **Perche' Postgres + pgvector**: offre un DB unico per metadati + ricerca vettoriale, senza
  introdurre un nuovo servizio.

### Componenti principali

- **Scanner** (`rag_indexer/scanner.py`):
  - percorre la directory root
  - applica include/exclude
  - scarta file binari o troppo grandi

- **Chunker** (`rag_indexer/chunking.py`):
  - Markdown: split per heading + fallback a chunk per lunghezza (con section_path)
  - Codice: chunk per lunghezza, con euristica per non spezzare funzioni/classi
  - Tipi trattati come codice: .py, .js, .ts, .cs, .aspx, .ashx
  - Output con offset e linee per citazioni

- **Embedder** (`rag_indexer/embedder.py`):
  - interfaccia `embed(text) -> list[float]`
  - implementazione locale con SentenceTransformers
  - (opzionale) implementazione Gemini per embedding remoti

- **Store** (`rag_indexer/store.py`):
  - v2: documents + chunks + embeddings
  - incremental ingest con hash e dedup

- **Retrieval** (`rag_indexer/retrieval.py`):
  - hybrid search (vector + keyword)
  - filtri per repo_id, doc_type, language, path_prefix
  - dedup e limite per documento

### Schema dati e indici (v2)
Lo schema v2 separa documenti e chunk per supportare filtri, versioning e citazioni line-based.

Tabelle principali:
- `documents` (repo_id, path_norm, content_hash, metadata)
- `chunks` (offset/linee, section_path, tsvector)
- `chunk_embeddings` (vettori + modello)

Indici chiave:
- btree su (repo_id, path_norm)
- GIN su `chunks.search_tsv`
- ivfflat su `chunk_embeddings.embedding`

Lo schema v2 e' l'unico schema supportato.

### Cosa fa l'indice ivfflat

`ivfflat` e' un indice approssimato: accelera la ricerca su grandi volumi riducendo il costo
di confronto tra vettori. Si sceglie un compromesso tra velocita' e accuratezza.
Il parametro `lists` controlla quante partizioni usare: piu' liste = piu' accuratezza ma piu' costo.

### Scelte tecniche principali

- **Postgres + pgvector**: soluzione semplice, robusta, self-hosted.
- **Incremental ingest (v2)**: aggiorna solo file modificati, dedup e mark-delete.
- **Embedding locali (384)**: niente dati inviati all'esterno, buone prestazioni su laptop.
- **Include/exclude mirati**: riduce rumore e dimensione dell'indice.
- **Scope controllato**: no root .asp, no App_Code, solo cartelle target.

### Perche' incremental ingest (v2)

Il v2 traccia hash e mtime: aggiorna solo i file cambiati e marca i file rimossi.
Riduce tempi di ingest e mantiene consistenza senza full reindex.

### Flusso logico (ingest v2)
1) scan dei file
2) lettura contenuto con controlli (size, binary)
3) normalizzazione + hash
4) chunking con offset/linee
5) embedding locale
6) upsert document + insert chunk + embedding

### Flusso logico (query v2)
1) embedding della query
2) ricerca ibrida (vector + keyword)
3) dedup/diversity + filtro per doc_type/language/path
4) ritorno snippet con citazioni line-based

### Limitazioni note
- La dimensione embedding e' fissa: se cambi modello devi ricreare la tabella.
- Chunking euristico (non parsing completo del codice).
- Reranker e' disabilitato di default (si usa solo hybrid + dedup).

### Impatto prestazioni (embedding locale)

- **RAM**: ~1-2 GB durante l'inferenza
- **CPU**: piu' lento di un'API remota, ma sufficiente per indicizzazione batch
- **Disco**: ~500MB per i pesi del modello

## Prerequisiti

- Python 3.10+
- Docker (per Postgres + pgvector)
- Connessione Internet solo per scaricare i pesi del modello locale al primo avvio

## Avvio Postgres + pgvector (Docker)

Se il container non esiste:

```bash
docker run --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg15
```

Se esiste gia':

```bash
docker start pgvector
```

## Installazione dipendenze

Dalla root del repo:

```bash
pip install -e llm_context
```

## Configurazione

File principale: `llm_context/config.yaml`.

Per mantenere i modelli locali dentro il progetto, usa cache locale:

```bash
set MCP_MODELS_DIR=.local/models
set HF_HOME=.local/models/huggingface
set TRANSFORMERS_CACHE=.local/models/huggingface/transformers
set SENTENCE_TRANSFORMERS_HOME=.local/models/huggingface/sentence_transformers

# MCP transport security / behavior
set MCP_HOST=127.0.0.1
set MCP_PORT=8765
set MCP_SSE_ENABLED=false
set MCP_ALLOWED_HOSTS=localhost,127.0.0.1,::1
set MCP_ALLOWED_ORIGINS=http://localhost:*,http://127.0.0.1:*,https://localhost:*,https://127.0.0.1:*
```

Valori chiave (gia' impostati per uso locale):

- `embedding_dim: 384`
- `include_dirs`: scope di indicizzazione (vedi sotto)
- `assets_template_only: true`
- `scope_map`: parole chiave -> path prefix per auto-scope nelle query

### Parametri principali (spiegazione)

- `embedding_dim`: dimensione dei vettori; deve combaciare col modello.
- `ivfflat_lists`: partizioni dell'indice vettoriale (piu' alto = piu' accurato/meno veloce).
- `chunk_size`, `chunk_overlap`: dimensione chunk e sovrapposizione per non spezzare concetti.
- `md_chunk_size`, `code_chunk_size`: taratura specifica per Markdown e codice.
- `include_dirs`: whitelist delle cartelle o pattern da indicizzare.
- `min_score`: soglia minima per accettare un chunk nel retrieval v2.
- `default_doc_type`: filtro predefinito per v2 (es. `code`).
- `vector_weight`, `keyword_weight`: pesi della fusione hybrid.

### Scope attuale di indicizzazione (v2)

Nel file `llm_context/config.yaml`:

```
include_dirs:
  - pubblico/api
  - pubblico/bpofh
  - pubblico/js
  - pubblico/css
  - pubblico/assets/template
  - pubblico/*.aspx
  - pubblico/*.html
  - librerie/BpoFH
  - librerie/Fiscobot
  - librerie/primanota
  - db/schema-completo
  - instructions
```

Nota: il matching e' case-insensitive.

### Exclude mirati

- `App_Code/**`
- `*.env*`, `web.config`, `pubblico/Web.config`
- `*.pfx`, `*.pem`, `*.key`, `*.cer`
- `node_modules`, `bin`, `obj`, `packages`, `Log`, `upload`, `allegati`

### Come modificare i default in futuro

Modifica `llm_context/config.yaml`:
- `default_doc_type`: usa `code`, `markdown`, `text`, `config` o `null` (nessun filtro).
- `min_score`: alza per ridurre rumore, abbassa per avere piu' recall.
- `vector_weight`/`keyword_weight`: bilancia semantica vs keyword (somma consigliata ~1.0).
- `max_chunks_per_doc`: limita il numero di chunk per documento.

Esempio:
```
default_doc_type: code
min_score: 0.25
vector_weight: 0.7
keyword_weight: 0.3
```

## Reset completo (drop + reinit v2)

Attenzione: questo cancella tutto l'indice esistente.

```bash
docker exec -it pgvector psql -U postgres -d postgres -c "DROP TABLE IF EXISTS chunk_embeddings; DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS documents;"
python llm_context/cli.py init-db-v2 --dsn "postgresql://postgres:postgres@localhost:5432/postgres" --embedding-dim 384
```

## Ingest v2 incrementale (locale, senza invio dati)

```bash
python llm_context/cli.py --verbose ingest --incremental \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --root ./ \
  --embedder local-st
```

### Modello embeddings locale

Di default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dimensione 384).
E' un modello multilingua, adatto a testi italiani, con footprint ridotto.

Se vuoi cambiare modello:

1. Aggiorna `embedding_dim` nel config
2. Ricrea la tabella
3. Usa `--local-model <nome_modello>`

## Query di test (v2)

```bash
python llm_context/cli.py query \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --text "come funziona X?" \
  --top-k 8 \
  --path-prefix "librerie\\BpoFH\\" \
  --embedder local-st
```

Oppure usa direttamente un file/dir per filtrare:

```bash
python llm_context/cli.py query \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --text "spiegami Attivita" \
  --top-k 8 \
  --file "librerie\\BpoFH\\Attivita.cs" \
  --embedder local-st
```

## Context pronto per agenti (CLI, v2)

Comando che restituisce solo il contesto da incollare nel prompt:

```bash
python llm_context/cli.py context \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --text "come funziona bpofh?" \
  --top-k 8 \
  --path-prefix "librerie\\BpoFH\\" \
  --embedder local-st
```

Auto-scope: se il testo contiene parole chiave (es. "bpofh", "fiscobot"),
il comando applica automaticamente un filtro sul path. La mappa e' in `config.yaml` (scope_map).
Se il testo contiene un path (es. `pubblico/assets/pdf/mandato_professionale_gallo.html`),
il tool lo rileva e applica il filtro automaticamente.

Regola operativa: prima di usare PROJECT_INDEX o grep, eseguire sempre almeno una query `context`.

Oppure usa direttamente un file/dir per filtrare:

```bash
python llm_context/cli.py context \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --text "come funziona?" \
  --top-k 8 \
  --file "librerie\\BpoFH\\" \
  --embedder local-st
```

## Troubleshooting rapido

### Errore dimensione embedding
Se vedi un errore tipo `expected 768 dimensions, not 384`, la tabella e' stata creata con un'altra dimensione.
Esegui il reset completo (drop + init).

### `psql` non riconosciuto
Usa `psql` dentro Docker:

```bash
docker exec -it pgvector psql -U postgres -d postgres -c "SELECT count(*) FROM chunks;"
```

## Note su privacy

Con `--embedder local-st` non c'e' invio dati a servizi esterni.
L'unica operazione online e' il download iniziale dei pesi del modello.

## Integrazione con agenti (uso DB per ricerche)

L'agente deve fare due step:
1) calcolare l'embedding della query
2) recuperare i chunk piu' simili da Postgres

Nel progetto trovi un helper pronto: `rag_indexer/agent_context.py`.

Esempio minimo (Python):

```python
from rag_indexer.agent_context import build_context
from rag_indexer.embedder import LocalSentenceTransformerEmbedder

DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
PROJECT_ID = "myproj"

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

Come usare il contesto in un prompt:

```
System: Sei un assistente tecnico.
User: Domanda utente...
Context:
<context generato>
```

Se vuoi usare un embedding esterno, puoi passare direttamente `query_embedding`
al posto di `query_text`.

Nota: `embedding_dim` deve corrispondere alla dimensione usata per il DB (384).

### Guida per agenti

Ho aggiunto un documento dedicato per l'uso da parte degli agenti:

- `llm_context/AGENT_GUIDE.md`

Contiene:
- esempio con helper `build_context()`
- query SQL diretta con psycopg
- linee guida su filtri e privacy

## Recap comandi (llm_context, v2)

Install:

```bash
pip install -e llm_context
```

Init DB v2:

```bash
python llm_context/cli.py init-db-v2 --dsn "postgresql://postgres:postgres@localhost:5432/postgres" --embedding-dim 384
```

Ingest v2 incrementale:

```bash
python llm_context/cli.py --verbose ingest --incremental \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --root ./ \
  --embedder local-st
```

Context (per agenti, v2):

```bash
python llm_context/cli.py context \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --text "stiamo lavorando su bpofh" \
  --top-k 8 \
  --embedder local-st
```

Query normale (v2):

```bash
python llm_context/cli.py query \
  --dsn "postgresql://postgres:postgres@localhost:5432/postgres" \
  --project-id myproj \
  --text "come funziona bpofh?" \
  --top-k 8 \
  --embedder local-st
```
