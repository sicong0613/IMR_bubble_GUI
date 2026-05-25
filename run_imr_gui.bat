@echo off
setlocal

set "VENV_DIR=C:\venvs\imr-gui"
set "PROJECT_DIR=%~dp0"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo IMR Fitting GUI virtual environment was not found:
    echo   %VENV_DIR%
    echo.
    echo Create it with:
    echo   py -3.11 -m venv %VENV_DIR%
    echo   %VENV_DIR%\Scripts\activate
    echo   cd /d "%PROJECT_DIR%"
    echo   python -m pip install --upgrade pip
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
"%VENV_DIR%\Scripts\python.exe" main.py

if errorlevel 1 (
    echo.
    echo IMR Fitting GUI exited with an error.
    pause
)
