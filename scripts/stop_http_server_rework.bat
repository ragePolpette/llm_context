@echo off
REM Ferma il runtime HTTP rework sul port configurato.

cd /d "%~dp0\.."

set PORT=8766
if not "%MCP_PORT%"=="" set PORT=%MCP_PORT%

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  echo [INFO] Termino PID %%p in ascolto su porta %PORT%
  taskkill /PID %%p /F >nul 2>&1
)

echo [INFO] Stop richiesto sulla porta %PORT%.
