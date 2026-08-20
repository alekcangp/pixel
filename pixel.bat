@echo off
rem ============================================================
rem  Pixel - unified install/run/uninstall script (Windows)
rem
rem  Usage:
rem    pixel.bat            install (if needed) + run
rem    pixel.bat install    install only (venv + deps + model)
rem    pixel.bat run        run only
rem    pixel.bat uninstall  uninstall (venv, DB, model cache, app files)
rem ============================================================
setlocal
cd /d "%~dp0"

set "SHOULD_RUN_AFTER_INSTALL=0"
if "%~1"=="" set "SHOULD_RUN_AFTER_INSTALL=1"

rem Detect working Python executable.
rem On Windows 10/11 the App Execution Alias may make `where python.exe`
rem succeed even when Python is not actually installed; verify with --version.
set "PYTHON_CMD="
for %%p in (python.exe python3.exe py.exe) do (
    where "%%p" >nul 2>&1
    if not errorlevel 1 (
        "%%p" --version >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_CMD=%%p"
            goto :PYTHON_FOUND
        )
    )
)
:PYTHON_FOUND

if "%PYTHON_CMD%"=="" (
    echo [Pixel] Python not found. Installing Python 3.12...
    call :AUTO_INSTALL_PYTHON
    if errorlevel 1 (
        echo [Pixel] Error: Python installation failed.
        pause
        exit /b 1
    )
    echo [Pixel] Python installed successfully.
    set "PYTHON_CMD="
    for %%p in (python.exe python3.exe py.exe) do (
        where "%%p" >nul 2>&1
        if not errorlevel 1 (
            "%%p" --version >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_CMD=%%p"
                goto :PYTHON_FOUND
            )
        )
    )
    if "%PYTHON_CMD%"=="" (
        echo [Pixel] Python installed, but not yet in PATH. Restart the terminal and run pixel.bat again.
        pause
        exit /b 1
    )
)

echo [Pixel] Using Python: %PYTHON_CMD%

if /i "%~1"=="install"   goto INSTALL
if /i "%~1"=="run"       goto RUN
if /i "%~1"=="uninstall" goto UNINSTALL

rem No argument: install (if venv missing) + run
if not exist ".venv\Scripts\python.exe" goto INSTALL
goto RUN

:INSTALL
echo [Pixel] Setting up virtual environment...
if not exist ".venv\Scripts\python.exe" (
    "%PYTHON_CMD%" -m venv .venv
    if errorlevel 1 goto FAIL
)
call ".venv\Scripts\activate.bat"

echo [Pixel] Installing dependencies...
"%PYTHON_CMD%" -m pip install --upgrade pip
"%PYTHON_CMD%" -m pip install -r requirements.txt
if errorlevel 1 goto FAIL

echo [Pixel] Downloading SigLIP model (first run)...
"%PYTHON_CMD%" -c "from transformers import AutoProcessor, AutoModel; m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m); print('Model loaded:', m)"
if errorlevel 1 goto FAIL

echo [Pixel] Setup complete.
if %SHOULD_RUN_AFTER_INSTALL%==1 goto RUN
pause
goto :EOF

:RUN
if not exist ".venv\Scripts\python.exe" (
    echo [Pixel] Environment not found. Run: pixel.bat install
    pause
    goto FAIL
)
echo [Pixel] Launching Pixel...
".venv\Scripts\python.exe" main.py ui-flet
goto :EOF

:UNINSTALL
echo [Pixel] Removing virtual environment...
if exist ".venv" rmdir /s /q ".venv"
echo [Pixel] Removing database storage\*.db...
if exist "storage" del /f /q "storage\*.db" "storage\*.db-journal" "storage\*.db-wal" "storage\*.db-shm" 2>nul
echo [Pixel] Removing HuggingFace model cache...
if exist "%USERPROFILE%\.cache\huggingface\hub\models--google--siglip-base-patch16-224" rmdir /s /q "%USERPROFILE%\.cache\huggingface\hub\models--google--siglip-base-patch16-224"
if exist "%USERPROFILE%\.cache\torch\transformers\google--siglip-base-patch16-224" rmdir /s /q "%USERPROFILE%\.cache\torch\transformers\google--siglip-base-patch16-224"
echo [Pixel] Removing application files...
set "APP_DIR=%~dp0"
start /b "" cmd /c "ping -n 4 127.0.0.1 >nul & rmdir /s /q "%APP_DIR%""
echo [Pixel] Uninstall complete.
pause
goto :EOF

:FAIL
echo [Pixel] Error. See output above.
pause
exit /b 1

:AUTO_INSTALL_PYTHON
set "PYTHON_INSTALLER=%TEMP%\python-3.12.9-amd64.exe"
echo [Pixel] Downloading Python 3.12.9 installer...
powershell -Command "& {Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing}"
if not exist "%PYTHON_INSTALLER%" (
    echo [Pixel] Failed to download Python installer.
    exit /b 1
)
echo [Pixel] Installing Python 3.12.9 silently...
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0
if errorlevel 1 (
    echo [Pixel] Python installer returned an error.
    exit /b 1
)
exit /b 0
