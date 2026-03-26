@echo off
REM Inizializza lo schema v2 sul target DSN del rework.

cd /d "%~dp0\.."

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERRORE] LLM_CONTEXT_DSN non impostato.
  echo [INFO] Passa il DSN al processo dal PowerShell di lancio oppure dalla dashboard MCP.
  exit /b 1
)

python cli.py --config config.rework.yaml init-db-v2 --dsn "%LLM_CONTEXT_DSN%" --embedding-dim 384
