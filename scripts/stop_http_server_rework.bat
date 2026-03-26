@echo off
REM Ferma il runtime HTTP rework sul port configurato.

cd /d "%~dp0\.."

set PORT=8766
if exist ".env.rework" (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$envFile = Join-Path (Get-Location) '.env.rework'; Get-Content $envFile | Where-Object { $_ -match '^MCP_PORT=' } | Select-Object -First 1 | ForEach-Object { ($_ -split '=',2)[1] }"`) do set PORT=%%i
)

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  echo [INFO] Termino PID %%p in ascolto su porta %PORT%
  taskkill /PID %%p /F >nul 2>&1
)

echo [INFO] Stop richiesto sulla porta %PORT%.
