@echo off
REM Esegue uno smoke test del runtime HTTP rework via /health e /rpc.

cd /d "%~dp0\.."

python scripts\smoke_test_rework_http.py %*
