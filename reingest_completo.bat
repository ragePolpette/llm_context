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
  echo [ERRORE] LLM_CONTEXT_DSN non impostata.
  exit /b 1
)
if "%LLM_CONTEXT_PROJECT_ID%"=="" set "LLM_CONTEXT_PROJECT_ID=myproj"
if "%LLM_CONTEXT_EMBEDDER%"=="" set "LLM_CONTEXT_EMBEDDER=local-st"
if "%LLM_CONTEXT_EMBEDDING_DIM%"=="" set "LLM_CONTEXT_EMBEDDING_DIM=384"
if "%LLM_CONTEXT_IVFFLAT_LISTS%"=="" set "LLM_CONTEXT_IVFFLAT_LISTS=100"
if "%LLM_CONTEXT_LOCAL_MODEL%"=="" set "LLM_CONTEXT_LOCAL_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
if "%LLM_CONTEXT_CONFIG%"=="" set "LLM_CONTEXT_CONFIG=%~dp0config.yaml"

if "%LLM_CONTEXT_ROOT%"=="" (
  for %%I in ("%~dp0..") do set "LLM_CONTEXT_ROOT=%%~fI"
)

if "%MCP_MODELS_DIR%"=="" set "MCP_MODELS_DIR=%CD%\.local\models"
if "%HF_HOME%"=="" set "HF_HOME=%MCP_MODELS_DIR%\huggingface"
if "%TRANSFORMERS_CACHE%"=="" set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"
if "%SENTENCE_TRANSFORMERS_HOME%"=="" set "SENTENCE_TRANSFORMERS_HOME=%HF_HOME%\sentence_transformers"
if "%HF_HUB_DISABLE_TELEMETRY%"=="" set "HF_HUB_DISABLE_TELEMETRY=1"

echo ============================================================
echo   Re-ingest Completo LLM Context
echo ============================================================
echo DSN:         %LLM_CONTEXT_DSN%
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
echo [2/2] Full ingest (non incrementale)...
%PYTHON_EXE% cli.py --config "%LLM_CONTEXT_CONFIG%" --verbose ingest ^
  --dsn "%LLM_CONTEXT_DSN%" ^
  --project-id "%LLM_CONTEXT_PROJECT_ID%" ^
  --root "%LLM_CONTEXT_ROOT%" ^
  --embedder "%LLM_CONTEXT_EMBEDDER%" ^
  --embedding-dim %LLM_CONTEXT_EMBEDDING_DIM% ^
  --local-model "%LLM_CONTEXT_LOCAL_MODEL%"
if errorlevel 1 goto :error

echo.
echo [OK] Re-ingest completo terminato.
exit /b 0

:dry_run
echo [DRY-RUN] Comandi che verrebbero eseguiti:
echo %PYTHON_EXE% cli.py --config "%LLM_CONTEXT_CONFIG%" init-db-v2 --dsn "%LLM_CONTEXT_DSN%" --embedding-dim %LLM_CONTEXT_EMBEDDING_DIM% --ivfflat-lists %LLM_CONTEXT_IVFFLAT_LISTS%
echo %PYTHON_EXE% cli.py --config "%LLM_CONTEXT_CONFIG%" --verbose ingest --dsn "%LLM_CONTEXT_DSN%" --project-id "%LLM_CONTEXT_PROJECT_ID%" --root "%LLM_CONTEXT_ROOT%" --embedder "%LLM_CONTEXT_EMBEDDER%" --embedding-dim %LLM_CONTEXT_EMBEDDING_DIM% --local-model "%LLM_CONTEXT_LOCAL_MODEL%"
exit /b 0

:help
echo Uso:
echo   reingest_completo.bat
echo.
echo Variabili ambiente opzionali:
echo   PYTHON                     eseguibile Python (default: python)
echo   LLM_CONTEXT_DSN            DSN Postgres
echo   LLM_CONTEXT_PROJECT_ID     project id da re-indicizzare
echo   LLM_CONTEXT_ROOT           root del codice da indicizzare
echo   LLM_CONTEXT_CONFIG         path al config.yaml
echo   LLM_CONTEXT_EMBEDDER       local-st ^| local-hash ^| dummy
echo   LLM_CONTEXT_LOCAL_MODEL    nome modello sentence-transformers
echo   LLM_CONTEXT_EMBEDDING_DIM  dimensione embedding (default: 384)
echo   LLM_CONTEXT_IVFFLAT_LISTS  liste indice ivfflat (default: 100)
echo   LLM_CONTEXT_DRY_RUN        1 per stampare i comandi senza eseguirli
echo.
echo Esempio:
echo   set LLM_CONTEXT_PROJECT_ID=myproj
echo   set LLM_CONTEXT_ROOT=C:\repo
echo   reingest_completo.bat
exit /b 0

:error
echo.
echo [ERRORE] Operazione fallita.
exit /b 1
