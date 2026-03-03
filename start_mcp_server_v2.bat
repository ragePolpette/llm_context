@echo off
echo ============================================================
echo   LLM Context MCP Server - Avvio (v2)
echo ============================================================
echo.

cd /d "%~dp0"

:: Imposta variabili d'ambiente
set LLM_CONTEXT_DSN=postgresql://postgres:postgres@localhost:5432/postgres
set LLM_CONTEXT_PROJECT_ID=myproj
set LLM_CONTEXT_EMBEDDER=local-st
set LLM_CONTEXT_EMBEDDING_DIM=384
set LLM_CONTEXT_LOCAL_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
set LLM_CONTEXT_MAX_CHARS=12000
set MCP_MODELS_DIR=%CD%\.local\models
set HF_HOME=%MCP_MODELS_DIR%\huggingface
set TRANSFORMERS_CACHE=%HF_HOME%\transformers
set SENTENCE_TRANSFORMERS_HOME=%HF_HOME%\sentence_transformers
set HF_HUB_OFFLINE=0
set TRANSFORMERS_OFFLINE=0
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONPATH=%CD%

echo [INFO] Schema v2 attivo.
echo [INFO] Avvio del server MCP...
echo [INFO] Una volta che vedi "Embedder ready!", puoi aprire il client MCP.
echo.

python -u mcp_server_http.py

pause
