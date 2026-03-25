@echo off
REM Avvia il runtime stdio del rework con profilo dedicato, separato dal servizio live.

echo ============================================
echo   LLM Context Rework Stdio Server
echo ============================================
echo.

cd /d "%~dp0\.."

if not exist ".env.rework" (
  copy /Y "config.local.example.rework.env" ".env.rework" >nul
  echo [INFO] Creato .env.rework dal template.
  echo [INFO] Aggiorna LLM_CONTEXT_DSN in .env.rework e rilancia lo script.
  exit /b 1
)

set LLM_CONTEXT_DOTENV_PATH=%CD%\.env.rework
if "%LLM_CONTEXT_CONFIG_PATH%"=="" set LLM_CONTEXT_CONFIG_PATH=%CD%\config.rework.yaml
if "%PYTHONPATH%"=="" set PYTHONPATH=%CD%

echo [INFO] Runtime profile: %LLM_CONTEXT_CONFIG_PATH%
echo [INFO] Env file: %LLM_CONTEXT_DOTENV_PATH%
echo [INFO] Avvio server stdio del rework...
echo.

python -u mcp_server.py

pause
