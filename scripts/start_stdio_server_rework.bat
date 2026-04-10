@echo off
REM Wrapper legacy: inoltra allo script canonico stdio di llm-context.
cd /d "%~dp0"
call start_stdio_server.bat
