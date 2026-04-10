@echo off
REM Ferma il runtime HTTP canonico di llm-context sulla porta configurata.

cd /d "%~dp0\.."

set PORT=8765
if not "%MCP_PORT%"=="" set PORT=%MCP_PORT%

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  echo [INFO] Termino PID %%p in ascolto su porta %PORT%
  taskkill /PID %%p /F >nul 2>&1
)

echo [INFO] Stop richiesto sulla porta %PORT%.
