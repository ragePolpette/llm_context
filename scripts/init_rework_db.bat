@echo off
REM Wrapper legacy: inoltra allo script canonico init-db di llm-context.
cd /d "%~dp0"
call init_db.bat
