@echo off
REM Create the v2 schema on the database referenced by LLM_CONTEXT_DSN.

cd /d "%~dp0\.."

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERROR] LLM_CONTEXT_DSN is not set.
  exit /b 1
)

python cli.py --config config.rework.yaml init-db-v2 --dsn "%LLM_CONTEXT_DSN%" --embedding-dim 384
