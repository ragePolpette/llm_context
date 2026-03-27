# LLM Memory Indexer (Postgres + pgvector) - Guida Operativa

Questo progetto e' destinato a vivere come **repository standalone** dentro il workspace DEV `Yetzirah`.
Il runtime locale, quando deployato, resta in `Binah\llm_context`.

Questa guida descrive come replicare l'installazione e l'indicizzazione locale che abbiamo configurato per BpoPilot.
L'obiettivo e' indicizzare solo un sottoinsieme di cartelle del progetto, usare embedding locali (nessun invio dati),
e mantenere un indice v2 con metadata ricchi e aggiornamenti incrementali.
Questo MCP serve solo per retrieval di contesto tecnico (codice/documenti): non e' un sistema di memoria operativa persistente.

## Multi-project mode

`llm-context` puo' ora lavorare in due modalita':

- `single-project`:
  - comportamento legacy
  - puo' usare `default_project_id` come fallback sicuro
- `multi-project`:
  - richiede `project_id` esplicito per `rag_context`, `rag_search`, `symbol_search`
  - espone discovery via MCP con `list_projects` e `get_project_info`

I flag principali sono in `config.yaml` / `config.example.yaml`:

- `multi_project_enabled`
- `default_project_id`
- `projects_registry_path`
- `projects_state_path`
- `ingest_enabled`
- `write_enabled`

## Project registry

Il registro progetti e' file-based e inizialmente semplice. Un esempio e' in
[projects.example.yaml](/C:/Users/Gianmarco/Urgewalt/Yetzirah/llm_context_rework/projects.example.yaml).

Ogni progetto puo' dichiarare almeno:

- `project_id`
- `display_name`
- `root_path`
- `include_profile`
- `retrieval_profile`
- `include_dirs`
- `exclude_globs`
- `ingest_enabled`
- `write_enabled`

Lo stato runtime di ingest viene salvato separatamente in `projects.state.json`, per esempio:

- `last_ingest_status`
- `last_ingest_started_at`
- `last_ingest_finished_at`
- `last_successful_ingest_at`
- `last_ingest_duration_sec`
- `index_version`
- `index_fingerprint`

Per ogni progetto viene inoltre mantenuto un manifest separato in `project_manifests/<project_id>.index_manifest.json`.
Il manifest serve per health e diagnostica operativa e include, tra gli altri:

- `last_ingest_status`
- `last_ingest_started_at`
- `last_ingest_completed_at`
- `indexed_documents`
- `indexed_chunks`
- `indexed_symbols`
- `config_fingerprint`
- `source_fingerprint`
- `store_target`

## Read-plane vs write-plane

Il piano MCP standard resta di sola retrieval:

- `rag_context`
- `rag_search`
- `symbol_search`
- `list_projects`
- `get_project_info`
- `context_info`

Ruoli operativi consigliati:

- `rag_context`: tool principale per lavorare sul codice; restituisce un package funzionale assemblato
- `rag_search`: tool di approfondimento/raw per ricerca mirata, debug e conferme, con gruppi per file e hint di investigazione
- `symbol_search`: tool di precisione per signature, linee esatte e disambiguazione simboli
- `context_info`: discovery del server, ruoli tool, limiti, quick start e guida decisionale per scegliere il tool giusto
- `map_work_item_to_codebase`: mapping strutturato richiesta funzionale -> area/repo/path

L'ingest non e' esposto come tool MCP standard.
L'ingest resta una capability operativa separata, attivabile solo a startup/config level con:

- `ingest_enabled=true`
  oppure
- `write_enabled=true`

## Ingest CLI schedulabile

La CLI e' il write-plane consigliato.

Comandi nuovi/rilevanti:

- `python cli.py --config config.yaml list-projects --json`
- `python cli.py --config config.yaml ingest --dsn <dsn> --project-id <project_id>`
- `python cli.py --config config.yaml ingest-enabled-projects --dsn <dsn>`

Comportamento:

- `ingest` valida il `project_id` contro il registry
- se il progetto esiste nel registry, risolve `root_path` e profilo base dal registry
- aggiorna `projects.state.json` con stato, durata e fingerprint indice
- aggiorna anche il manifest per-progetto con conteggi indice, target storage e fingerprint operativi
- `ingest-enabled-projects` esegue batch solo sui progetti con `ingest_enabled=true`

Questo modello e' pensato per essere schedulato esternamente senza esporre l'ingest come capability MCP always-on.

## Health operativo

L'endpoint `/health` espone ora anche stato operativo utile per la dashboard esterna:

- `status`
- `database_runtime` con reachability del DSN, presenza `pgvector` e verifica schema v2
- `runtime_readiness` con verdetto sintetico (`ready`, `degraded`, `blocked`), reasoning e azioni consigliate
- `multi_project_enabled`
- `ingest_enabled`
- `write_enabled`
- `project_count`
- riepilogo per progetto con stato ingest e freshness base
- `project_manifest_dir`
- riepilogo manifest per progetto con conteggi indice e ultimo stato ingest
- stato `integrity` per progetto (`ok`, `indexing`, `not_indexed`, `stale`, `unreliable`) con motivi espliciti

Nel modello "bring your own PostgreSQL", la readiness non si limita piu' allo stato logico dei progetti:
- verifica che il database configurato via `LLM_CONTEXT_DSN` sia raggiungibile
- verifica che l'estensione `pgvector` sia installata
- verifica che lo schema v2 (`documents`, `chunks`, `chunk_embeddings`, `index_runs`, `symbols`) sia presente
- classifica il target DSN con `network_scope` / `deployment_hint` per distinguere meglio:
  - loopback locale o Docker con port mapping
  - alias Docker/interna tipo `host.docker.internal`
  - Postgres remoto o managed

Le `recommended_actions` di `runtime_readiness` usano questi hint per dare runbook piu' specifici al dashboard:
- check locale o container con port mapping se il DSN punta a `localhost` / `127.0.0.1`
- check rete Docker se il DSN usa alias container/host Docker
- check rete, firewall/VPN e `sslmode` se il DSN punta a Postgres remoto o cloud

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
- PostgreSQL con estensione `pgvector` disponibile in locale o su rete interna
- Connessione Internet solo per scaricare i pesi del modello locale al primo avvio

## Database: bring your own PostgreSQL

Il processo `llm-context` deve essere gestito come gli altri MCP:

- processo Python normale
- start/stop da dashboard o script locali
- database collegato a runtime
- DSN passato via env/runtime

Il servizio **non provisiona** il database e non assume un modello infrastrutturale specifico.
Il prodotto da esporre e' quindi: `collega il tuo PostgreSQL`.

Modalita' supportate:

- PostgreSQL locale installato sulla macchina
- PostgreSQL interno o remoto/cloud
- PostgreSQL in Docker, ma **solo come scelta di provisioning del DB**

Docker quindi resta opzionale e non fa parte del contratto operativo del servizio.
Il runtime MCP vede solo un DSN PostgreSQL valido.

## Database locale: requisiti minimi

Serve un PostgreSQL raggiungibile dal DSN e con:

- estensione `vector`
- estensione `unaccent`
- permessi per creare tabelle/indici dello schema v2

Esempio DSN:

```bash
postgresql://<user>:<password>@<host>:5432/<database>
```

## Credenziali e segreti

Le credenziali DB devono essere fornite **solo a runtime**:

- dal PowerShell di lancio
- oppure dal `mcp-dashboard`

Regole operative:

- nessun `.env`
- nessun DSN in `config.yaml` o `config.rework.yaml`
- nessuna credenziale persistita nel repo
- `/health` e `context_info` possono esporre solo dati safe come `host`, `database` e `dsn_fingerprint`

Nel dashboard la configurazione corretta e':

- un'opzione secret `LLM_CONTEXT_DSN`
- opzioni non secret per `LLM_CONTEXT_CONFIG_PATH`, `LLM_CONTEXT_RUNTIME_NAME`, `MCP_PORT`
- opzionalmente `LLM_CONTEXT_EMBEDDER` e `LLM_CONTEXT_STORE_TARGET`

## Installazione dipendenze

Dalla root del repo:

```bash
pip install -e .
```

## Configurazione

File principale: `config.yaml`.

Profilo runtime dedicato del rework:

- `config.rework.yaml`: profilo separato del rework, basato su `config.yaml`
- `projects.rework.yaml`: registry locale del rework
- nessun file `.env`: il runtime rework legge solo variabili gia' presenti nel processo

Questo profilo esiste per allineare `llm_context_rework` agli altri MCP del workspace:

- runtime Python normale
- config/env/porta separati dal live
- nessuna condivisione del runtime con `llm_context` attuale
- `runtime_name` esplicito per riconoscere il server rework da `/health` e `context_info`

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
set LLM_CONTEXT_HTTP_LOG_PATH=logs/mcp_server_http.log
set LLM_CONTEXT_HTTP_LOG_MAX_BYTES=1048576
set LLM_CONTEXT_HTTP_LOG_BACKUP_COUNT=3
set LLM_CONTEXT_MAX_QUERY_EMBEDDING_ITEMS=4096
set LLM_CONTEXT_MAX_REQUEST_BYTES=1048576
set LLM_CONTEXT_MAX_REQUEST_BYTES=1048576
set LLM_CONTEXT_MAX_QUERY_EMBEDDING_ITEMS=4096
```

Valori chiave (gia' impostati per uso locale):

- `embedding_dim: 384`
- `include_dirs`: scope di indicizzazione (vedi sotto)
- `assets_template_only: true`
- `scope_map`: parole chiave -> path prefix per auto-scope nelle query
- `LLM_CONTEXT_HTTP_LOG_PATH`: file log del server HTTP con rotazione
- `LLM_CONTEXT_HTTP_LOG_MAX_BYTES`: dimensione massima di ogni file log prima della rotazione
- `LLM_CONTEXT_HTTP_LOG_BACKUP_COUNT`: numero di file log storici mantenuti
- `LLM_CONTEXT_MAX_QUERY_EMBEDDING_ITEMS`: limite massimo di elementi accettati in `query_embedding`
- `LLM_CONTEXT_MAX_REQUEST_BYTES`: limite massimo del body JSON HTTP prima del parse
- `LLM_CONTEXT_MAX_REQUEST_BYTES`: limite massimo del body JSON accettato dagli endpoint HTTP prima del parse
- `LLM_CONTEXT_MAX_QUERY_EMBEDDING_ITEMS`: limite massimo di elementi accettati in `query_embedding`

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

Nel file `config.yaml`:

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

Modifica `config.yaml`:
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

Il DB puo' essere locale, remoto/cloud o in Docker; il servizio MCP non cambia.
Usa sempre il DSN del database che hai collegato al runtime.

```bash
psql "postgresql://<user>:<password>@<host>:5432/<database>" -c "DROP TABLE IF EXISTS chunk_embeddings; DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS documents;"
python cli.py init-db-v2 --dsn "postgresql://<user>:<password>@<host>:5432/<database>" --embedding-dim 384
```

## Avvio del runtime rework

Per avviare il rework come servizio separato:

1. imposta il `LLM_CONTEXT_DSN` dedicato del rework nel PowerShell di lancio oppure dalla dashboard MCP in `C:\Users\Gianmarco\Urgewalt\Yetzirah\mcp-dashboard`
2. opzionalmente imposta `MCP_PORT` e `LLM_CONTEXT_RUNTIME_NAME`
3. avvia `scripts/start_http_server_rework.bat`

Se il servizio e' orchestrato dal dashboard, il comportamento corretto e':

- il dashboard passa il `LLM_CONTEXT_DSN` come segreto runtime
- il dashboard passa `LLM_CONTEXT_CONFIG_PATH=config.rework.yaml`
- il dashboard passa `MCP_PORT=8766`
- il rework parte senza assumere nulla su Docker, localhost o provisioning del DB

Esempio PowerShell:

```powershell
$env:LLM_CONTEXT_DSN = "postgresql://<user>:<password>@<host>:5432/<database>"
$env:MCP_PORT = "8766"
$env:LLM_CONTEXT_RUNTIME_NAME = "rework"
.\scripts\start_http_server_rework.bat
```

Lo script usa:

- `config.rework.yaml`
- `projects.rework.yaml`
- porta dedicata via variabile `MCP_PORT` (default consigliato: `8766`)

Script operativi del runtime rework:

- `scripts/start_http_server_rework.bat`
- `scripts/start_stdio_server_rework.bat`
- `scripts/init_rework_db.bat`
- `scripts/ingest_rework.bat`
- `scripts/rework_status.bat`
- `scripts/stop_http_server_rework.bat`

## Smoke test del runtime rework

Dopo aver impostato il `LLM_CONTEXT_DSN` nel processo, puoi verificare il server HTTP del rework con:

```bash
scripts\smoke_test_rework_http.bat
```

Lo smoke test:

- avvia `mcp_server_http.py` con il profilo rework
- aspetta `/health`
- verifica `tools/list`
- esegue `context_info`, `list_projects` e `get_project_info` via `/rpc`
- verifica che `storage_target` sia visibile nel payload di health

Per esercitare anche i tool di retrieval:

```bash
scripts\smoke_test_rework_http.bat --exercise-retrieval
```

## Ingest v2 incrementale (locale, senza invio dati)

```bash
python cli.py --verbose ingest --incremental \
  --dsn "postgresql://<user>:<password>@localhost:5432/postgres" \
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
python cli.py query \
  --dsn "postgresql://<user>:<password>@localhost:5432/postgres" \
  --project-id myproj \
  --text "come funziona X?" \
  --top-k 8 \
  --path-prefix "librerie\\BpoFH\\" \
  --embedder local-st
```

Oppure usa direttamente un file/dir per filtrare:

```bash
python cli.py query \
  --dsn "postgresql://<user>:<password>@localhost:5432/postgres" \
  --project-id myproj \
  --text "spiegami Attivita" \
  --top-k 8 \
  --file "librerie\\BpoFH\\Attivita.cs" \
  --embedder local-st
```

## Context pronto per agenti (CLI, v2)

Comando che restituisce solo il contesto da incollare nel prompt:

```bash
python cli.py context \
  --dsn "postgresql://<user>:<password>@localhost:5432/postgres" \
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
python cli.py context \
  --dsn "postgresql://<user>:<password>@localhost:5432/postgres" \
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
Usa `psql` sul database locale o remoto raggiungibile dal DSN:

```bash
psql "postgresql://<user>:<password>@localhost:5432/postgres" -c "SELECT count(*) FROM chunks;"
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

DSN = "postgresql://<user>:<password>@localhost:5432/postgres"
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

- [AGENT_GUIDE.md](/C:/Users/Gianmarco/Urgewalt/Yetzirah/llm_context_rework/AGENT_GUIDE.md)

Contiene:
- esempio con helper `build_context()`
- query SQL diretta con psycopg
- linee guida su filtri e privacy

## Recap comandi (llm_context_rework, v2)

Install:

```bash
pip install -e .
```

Init DB v2:

```bash
python cli.py init-db-v2 --dsn "postgresql://<user>:<password>@localhost:5432/postgres" --embedding-dim 384
```

Ingest v2 incrementale:

```bash
python cli.py --verbose ingest --incremental \
  --dsn "postgresql://<user>:<password>@localhost:5432/postgres" \
  --project-id myproj \
  --root ./ \
  --embedder local-st
```

Context (per agenti, v2):

```bash
python cli.py context \
  --dsn "postgresql://<user>:<password>@localhost:5432/postgres" \
  --project-id myproj \
  --text "stiamo lavorando su bpofh" \
  --top-k 8 \
  --embedder local-st
```

Query normale (v2):

```bash
python cli.py query \
  --dsn "postgresql://<user>:<password>@localhost:5432/postgres" \
  --project-id myproj \
  --text "come funziona bpofh?" \
  --top-k 8 \
  --embedder local-st
```

