@echo off
REM Wrapper legacy: inoltra allo script canonico HTTP di llm-context.
cd /d "%~dp0"
call scripts\start_http_server.bat
