@echo off
REM Esegue ingest sul progetto canonico di llm-context.

cd /d "%~dp0\.."

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERRORE] LLM_CONTEXT_DSN non impostato.
  echo [INFO] Passa il DSN al processo dal PowerShell di lancio oppure dalla dashboard MCP.
  exit /b 1
)

set "PROJECT_ID=%LLM_CONTEXT_PROJECT_ID%"
if "%PROJECT_ID%"=="" (
  if not "%~1"=="" (
    set "PROJECT_ID=%~1"
    shift
  )
)

if "%PROJECT_ID%"=="" (
  echo [ERRORE] project_id non impostato.
  echo [INFO] Passa il project_id come primo argomento oppure imposta LLM_CONTEXT_PROJECT_ID nel processo.
  exit /b 1
)

python cli.py --config config.ingest.yaml ingest --dsn "%LLM_CONTEXT_DSN%" --project-id "%PROJECT_ID%" --embedder local-st --incremental %*
