@echo off
REM Avvia il runtime stdio del rework con profilo dedicato, separato dal servizio live.

echo ============================================
echo   LLM Context Rework Stdio Server
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
if "%LLM_CONTEXT_RUNTIME_NAME%"=="" set LLM_CONTEXT_RUNTIME_NAME=rework

echo [INFO] Runtime profile: %LLM_CONTEXT_CONFIG_PATH%
echo [INFO] Runtime name: %LLM_CONTEXT_RUNTIME_NAME%
echo [INFO] Avvio server stdio del rework...
echo.

python -u mcp_server.py

pause
