@echo off
REM Demo script for Repo-Trace (Windows version)
REM Starts the backend server and provides instructions for opening the UI

echo ==========================================
echo   Repo-Trace Demo
echo ==========================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo X Python is not installed or not in PATH
    exit /b 1
)

echo + Using Python: python
echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install -q -r requirements.txt
echo + Dependencies installed
echo.

REM Start the backend server
echo ==========================================
echo   Starting Backend Server
echo ==========================================
echo.
echo Backend will run on: http://127.0.0.1:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ----------------------------------------
echo   To use the UI:
echo ----------------------------------------
echo 1. Open frontend\index.html in your browser
echo    OR
echo 2. Run: start frontend\index.html
echo.
echo ----------------------------------------
echo   API Endpoints:
echo ----------------------------------------
echo * Health:  GET  http://127.0.0.1:8000/health
echo * Ingest:  POST http://127.0.0.1:8000/ingest
echo * Status:  GET  http://127.0.0.1:8000/status/{repo_id}
echo * Ask:     POST http://127.0.0.1:8000/ask
echo.
echo ==========================================
echo.

REM Start uvicorn
python -m uvicorn app.main:app --reload
