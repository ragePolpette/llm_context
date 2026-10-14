@echo off
REM Start the llm-context HTTP runtime.

cd /d "%~dp0\.."

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERROR] LLM_CONTEXT_DSN is not set.
  exit /b 1
)

if "%LLM_CONTEXT_CONFIG_PATH%"=="" set "LLM_CONTEXT_CONFIG_PATH=%CD%\config.rework.yaml"
if "%PYTHONPATH%"=="" set "PYTHONPATH=%CD%"
if "%LLM_CONTEXT_HTTP_LOG_PATH%"=="" set "LLM_CONTEXT_HTTP_LOG_PATH=%CD%\.local\logs\llm_context_http.log"
if "%LLM_CONTEXT_RUNTIME_NAME%"=="" set "LLM_CONTEXT_RUNTIME_NAME=llm-context"
if "%MCP_PORT%"=="" set "MCP_PORT=8765"

echo [INFO] Config: %LLM_CONTEXT_CONFIG_PATH%
echo [INFO] Runtime name: %LLM_CONTEXT_RUNTIME_NAME%
echo [INFO] MCP port: %MCP_PORT%

python -u mcp_server_http.py
