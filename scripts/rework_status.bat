@echo off
REM Mostra lo stato operativo del runtime HTTP rework.

cd /d "%~dp0\.."

if not exist ".env.rework" (
  echo [ERRORE] Manca .env.rework.
  exit /b 1
)

powershell -NoProfile -Command ^
  "$envFile = Join-Path (Get-Location) '.env.rework';" ^
  "$portLine = Get-Content $envFile | Where-Object { $_ -match '^MCP_PORT=' } | Select-Object -First 1;" ^
  "$port = if ($portLine) { ($portLine -split '=',2)[1] } else { '8766' };" ^
  "$uri = 'http://127.0.0.1:' + $port + '/health';" ^
  "Invoke-RestMethod -Uri $uri -Method Get | ConvertTo-Json -Depth 6"
