@echo off
REM Avvia il runtime HTTP canonico di llm-context.

echo ============================================
echo   LLM Context HTTP Server
echo ============================================
echo.

cd /d "%~dp0\.."

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERRORE] LLM_CONTEXT_DSN non impostato.
  echo [INFO] Passa il DSN al processo dal PowerShell di lancio oppure dalla dashboard MCP.
  exit /b 1
)

if "%LLM_CONTEXT_CONFIG_PATH%"=="" set LLM_CONTEXT_CONFIG_PATH=%CD%\config.rework.yaml
if "%PYTHONPATH%"=="" set PYTHONPATH=%CD%
if "%LLM_CONTEXT_HTTP_LOG_PATH%"=="" set LLM_CONTEXT_HTTP_LOG_PATH=%CD%\.local\logs\llm_context_http.log
if "%LLM_CONTEXT_RUNTIME_NAME%"=="" set LLM_CONTEXT_RUNTIME_NAME=llm-context
if "%MCP_PORT%"=="" set MCP_PORT=8765

echo [INFO] Runtime profile: %LLM_CONTEXT_CONFIG_PATH%
echo [INFO] Runtime name: %LLM_CONTEXT_RUNTIME_NAME%
echo [INFO] MCP port: %MCP_PORT%
echo [INFO] Avvio server HTTP di llm-context...
echo.

python -u mcp_server_http.py

pause
