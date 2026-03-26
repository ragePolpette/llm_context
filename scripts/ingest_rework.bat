@echo off
REM Esegue ingest sul progetto dedicato del rework.

cd /d "%~dp0\.."

if not exist ".env.rework" (
  echo [ERRORE] Manca .env.rework. Crea il file dal template prima dell'ingest.
  exit /b 1
)

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$envFile = Join-Path (Get-Location) '.env.rework'; Get-Content $envFile | Where-Object { $_ -match '^LLM_CONTEXT_DSN=' } | Select-Object -First 1 | ForEach-Object { ($_ -split '=',2)[1] }"`) do set LLM_CONTEXT_DSN=%%i

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERRORE] LLM_CONTEXT_DSN non trovato in .env.rework.
  exit /b 1
)

python cli.py --config config.rework.yaml ingest --dsn "%LLM_CONTEXT_DSN%" --project-id llm_context_rework --embedder local-st --incremental %*
