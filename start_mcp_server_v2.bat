@echo off
REM Wrapper legacy: inoltra allo script dedicato del rework.
cd /d "%~dp0"
call scripts\start_http_server_rework.bat
