@echo off
rem Starts the grading web app. Double-click this file, then open the address shown on the PC or phone (same Wi-Fi).
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
echo.
echo  Close this window to stop. Live scanning on the phone uses the https address printed below.
%PY% scripts\serve.py
pause
