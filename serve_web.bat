@echo off
REM One-click web server + tunnel. Double-click to start; "serve_web.bat stop" to stop.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\serve_web.ps1" %*
pause
