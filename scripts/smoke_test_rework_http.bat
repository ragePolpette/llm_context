@echo off
REM Esegue uno smoke test del runtime HTTP rework via /health e /rpc.

cd /d "%~dp0\.."

if not exist ".env.rework" (
  echo [ERRORE] Manca .env.rework. Crea il file dal template prima di eseguire lo smoke test.
  exit /b 1
)

python scripts\smoke_test_rework_http.py %*
