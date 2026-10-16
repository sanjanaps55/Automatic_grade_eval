@echo off
rem Starts the grading web app. Double-click this file, then open the address shown on the PC or phone (same Wi-Fi).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment .venv not found. See README.md "Setup".
  pause & exit /b 1
)
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do set IP=%%a
echo.
echo  On this PC:   http://localhost:8000
echo  On the phone: http://%IP: =%:8000   (same Wi-Fi)
echo  Close this window to stop.
echo.
.venv\Scripts\python -m uvicorn grade_eval.server:app --host 0.0.0.0 --port 8000
pause
