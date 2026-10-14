@echo off
REM Print the /health payload of the running HTTP runtime.

if "%MCP_PORT%"=="" set "MCP_PORT=8765"

powershell -NoProfile -Command ^
  "$headers = @{};" ^
  "if ($env:MCP_AUTH_TOKEN) { $headers['Authorization'] = 'Bearer ' + $env:MCP_AUTH_TOKEN };" ^
  "$uri = 'http://127.0.0.1:' + $env:MCP_PORT + '/health';" ^
  "Invoke-RestMethod -Uri $uri -Method Get -Headers $headers | ConvertTo-Json -Depth 6"
