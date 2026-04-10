@echo off
REM Wrapper legacy: inoltra allo script canonico di ingest di llm-context.
cd /d "%~dp0"
call ingest.bat %*
