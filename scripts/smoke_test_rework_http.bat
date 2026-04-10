@echo off
REM Wrapper legacy: inoltra allo smoke test HTTP canonico di llm-context.
cd /d "%~dp0\.."
call scripts\smoke_test_http.bat %*

