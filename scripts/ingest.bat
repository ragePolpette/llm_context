@echo off
REM Esegue ingest sul progetto canonico di llm-context.

cd /d "%~dp0\.."

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERRORE] LLM_CONTEXT_DSN non impostato.
  echo [INFO] Passa il DSN al processo dal PowerShell di lancio oppure dalla dashboard MCP.
  exit /b 1
)

python cli.py --config config.rework.yaml ingest --dsn "%LLM_CONTEXT_DSN%" --project-id llm_context --embedder local-st --incremental %*
