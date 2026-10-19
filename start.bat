@echo off
rem Starts the grading web app. Double-click this file, then open the address shown on the PC or phone (same Wi-Fi).
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
%PY% -c "import cv2, torch" >nul 2>&1
if errorlevel 1 if exist C:\Miniconda3\python.exe set PY=C:\Miniconda3\python.exe
echo.
echo  Close this window to stop. Open the phone address printed below to upload photos from the phone.
%PY% scripts\serve.py
pause
