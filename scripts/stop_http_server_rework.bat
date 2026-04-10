@echo off
REM Wrapper legacy: inoltra allo script canonico stop HTTP di llm-context.
cd /d "%~dp0"
call stop_http_server.bat
