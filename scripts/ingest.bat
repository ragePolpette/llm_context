@echo off
REM Incremental ingest of a registered project. Usage: ingest.bat [project_id] [extra cli args]

cd /d "%~dp0\.."

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERROR] LLM_CONTEXT_DSN is not set.
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
  echo [ERROR] project_id is not set. Pass it as the first argument or set LLM_CONTEXT_PROJECT_ID.
  exit /b 1
)

python cli.py --config config.ingest.yaml ingest --dsn "%LLM_CONTEXT_DSN%" --project-id "%PROJECT_ID%" --embedder local-st --incremental %*
