@echo off
REM Start the llm-context stdio runtime (newline-delimited JSON-RPC on stdin/stdout).

cd /d "%~dp0\.."

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERROR] LLM_CONTEXT_DSN is not set. 1>&2
  exit /b 1
)

if "%LLM_CONTEXT_CONFIG_PATH%"=="" set "LLM_CONTEXT_CONFIG_PATH=%CD%\config.rework.yaml"
if "%PYTHONPATH%"=="" set "PYTHONPATH=%CD%"
if "%LLM_CONTEXT_RUNTIME_NAME%"=="" set "LLM_CONTEXT_RUNTIME_NAME=llm-context"

python -u mcp_server.py %*
