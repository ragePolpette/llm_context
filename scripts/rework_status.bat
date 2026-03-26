@echo off
REM Mostra lo stato operativo del runtime HTTP rework.

cd /d "%~dp0\.."

if "%MCP_PORT%"=="" set MCP_PORT=8766

powershell -NoProfile -Command ^
  "$uri = 'http://127.0.0.1:' + $env:MCP_PORT + '/health';" ^
  "Invoke-RestMethod -Uri $uri -Method Get | ConvertTo-Json -Depth 6"
