@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if /I "%~1"=="--help" goto :help

if "%PYTHON%"=="" (
  set "PYTHON_EXE=python"
) else (
  set "PYTHON_EXE=%PYTHON%"
)

if "%LLM_CONTEXT_DSN%"=="" (
  echo [ERROR] LLM_CONTEXT_DSN is not set.
  exit /b 1
)
if "%LLM_CONTEXT_PROJECT_ID%"=="" set "LLM_CONTEXT_PROJECT_ID=llm_context"
if "%LLM_CONTEXT_EMBEDDER%"=="" set "LLM_CONTEXT_EMBEDDER=local-st"
if "%LLM_CONTEXT_EMBEDDING_DIM%"=="" set "LLM_CONTEXT_EMBEDDING_DIM=384"
if "%LLM_CONTEXT_IVFFLAT_LISTS%"=="" set "LLM_CONTEXT_IVFFLAT_LISTS=100"
if "%LLM_CONTEXT_LOCAL_MODEL%"=="" set "LLM_CONTEXT_LOCAL_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
if "%LLM_CONTEXT_CONFIG%"=="" set "LLM_CONTEXT_CONFIG=%~dp0config.ingest.yaml"
if "%LLM_CONTEXT_ROOT%"=="" set "LLM_CONTEXT_ROOT=%~dp0"
if "%LLM_CONTEXT_ROOT:~-1%"=="\" set "LLM_CONTEXT_ROOT=%LLM_CONTEXT_ROOT:~0,-1%"

if "%MCP_MODELS_DIR%"=="" set "MCP_MODELS_DIR=%CD%\.local\models"
if "%HF_HOME%"=="" set "HF_HOME=%MCP_MODELS_DIR%\huggingface"
if "%TRANSFORMERS_CACHE%"=="" set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"
if "%SENTENCE_TRANSFORMERS_HOME%"=="" set "SENTENCE_TRANSFORMERS_HOME=%HF_HOME%\sentence_transformers"
if "%HF_HUB_DISABLE_TELEMETRY%"=="" set "HF_HUB_DISABLE_TELEMETRY=1"

echo ============================================================
echo   llm-context full re-ingest
echo ============================================================
echo CONFIG:      %LLM_CONTEXT_CONFIG%
echo PROJECT_ID:  %LLM_CONTEXT_PROJECT_ID%
echo ROOT:        %LLM_CONTEXT_ROOT%
echo EMBEDDER:    %LLM_CONTEXT_EMBEDDER%
echo MODEL:       %LLM_CONTEXT_LOCAL_MODEL%
echo.

if /I "%LLM_CONTEXT_DRY_RUN%"=="1" goto :dry_run

echo [1/2] Init schema v2...
%PYTHON_EXE% cli.py --config "%LLM_CONTEXT_CONFIG%" init-db-v2 ^
  --dsn "%LLM_CONTEXT_DSN%" ^
  --embedding-dim %LLM_CONTEXT_EMBEDDING_DIM% ^
  --ivfflat-lists %LLM_CONTEXT_IVFFLAT_LISTS%
if errorlevel 1 goto :error

echo.
echo [2/2] Full (non-incremental) ingest...
%PYTHON_EXE% cli.py --config "%LLM_CONTEXT_CONFIG%" --verbose ingest ^
  --dsn "%LLM_CONTEXT_DSN%" ^
  --project-id "%LLM_CONTEXT_PROJECT_ID%" ^
  --root "%LLM_CONTEXT_ROOT%" ^
  --embedder "%LLM_CONTEXT_EMBEDDER%" ^
  --embedding-dim %LLM_CONTEXT_EMBEDDING_DIM% ^
  --local-model "%LLM_CONTEXT_LOCAL_MODEL%"
if errorlevel 1 goto :error

echo.
echo [OK] Full re-ingest completed.
exit /b 0

:dry_run
echo [DRY-RUN] Commands that would run (DSN redacted):
echo %PYTHON_EXE% cli.py --config "%LLM_CONTEXT_CONFIG%" init-db-v2 --dsn "***" --embedding-dim %LLM_CONTEXT_EMBEDDING_DIM% --ivfflat-lists %LLM_CONTEXT_IVFFLAT_LISTS%
echo %PYTHON_EXE% cli.py --config "%LLM_CONTEXT_CONFIG%" --verbose ingest --dsn "***" --project-id "%LLM_CONTEXT_PROJECT_ID%" --root "%LLM_CONTEXT_ROOT%" --embedder "%LLM_CONTEXT_EMBEDDER%" --embedding-dim %LLM_CONTEXT_EMBEDDING_DIM% --local-model "%LLM_CONTEXT_LOCAL_MODEL%"
exit /b 0

:help
echo Usage:
echo   reingest_full.bat
echo.
echo Optional environment variables:
echo   PYTHON                     Python executable (default: python)
echo   LLM_CONTEXT_DSN            PostgreSQL DSN (required, never printed)
echo   LLM_CONTEXT_PROJECT_ID     project id to re-index (default: llm_context)
echo   LLM_CONTEXT_ROOT           root of the code to index (default: this folder)
echo   LLM_CONTEXT_CONFIG         config file (default: config.ingest.yaml)
echo   LLM_CONTEXT_EMBEDDER       local-st ^| local-hash
echo   LLM_CONTEXT_LOCAL_MODEL    sentence-transformers model name
echo   LLM_CONTEXT_EMBEDDING_DIM  embedding dimension (default: 384)
echo   LLM_CONTEXT_IVFFLAT_LISTS  ivfflat index lists (default: 100)
echo   LLM_CONTEXT_DRY_RUN        1 to print the commands without running them
exit /b 0

:error
echo.
echo [ERROR] Operation failed.
exit /b 1
