@echo off
setlocal
cd /d "%~dp0"

rem =============================================================================
rem AI-Driven Phishing Email Detection Using NLP
rem One-Click Windows Launcher
rem
rem Double-click this file to start the application. The Streamlit server is
rem started headlessly, the default browser opens automatically, and all log
rem output is written to logs\app_streamlit.log.
rem =============================================================================

rem Prefer the project virtual environment, fall back to any Python on PATH.
set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

title AI Phishing Shield

"%PYTHON%" "%~dp0run.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo The application could not be started.
    echo See logs\app_streamlit.log for details.
    pause
)

endlocal & exit /b %EXIT_CODE%
