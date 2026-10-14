@echo off
REM Stop the HTTP runtime started from this checkout, using the PID it recorded at startup.

cd /d "%~dp0\.."

if not exist ".mcp_server.pid" (
  echo [INFO] No PID file found: the server is not running from this checkout.
  exit /b 0
)

set /p SERVER_PID=<".mcp_server.pid"
if "%SERVER_PID%"=="" (
  echo [ERROR] PID file is empty.
  exit /b 1
)

tasklist /FI "PID eq %SERVER_PID%" /FI "IMAGENAME eq python*" | find "%SERVER_PID%" >nul
if errorlevel 1 (
  echo [INFO] PID %SERVER_PID% is not a running Python process; removing stale PID file.
  del ".mcp_server.pid"
  exit /b 0
)

echo [INFO] Stopping PID %SERVER_PID% recorded in .mcp_server.pid
taskkill /PID %SERVER_PID% /T /F
