@echo off
rem ============================================================
rem  Pixel - unified install/run/uninstall script (Windows)
rem
rem  Usage:
rem    pixel.bat            install (if needed) + run
rem    pixel.bat install    install only (venv + deps + model)
rem    pixel.bat run        run only
rem    pixel.bat uninstall  uninstall (venv, DB, model cache)
rem ============================================================
setlocal
cd /d "%~dp0"

if /i "%~1"=="install"   goto INSTALL
if /i "%~1"=="run"       goto RUN
if /i "%~1"=="uninstall" goto UNINSTALL

rem No argument: install (if venv missing) + run
if not exist ".venv\Scripts\python.exe" goto INSTALL
goto RUN

:INSTALL
echo [Pixel] Setting up virtual environment...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto FAIL
)
call ".venv\Scripts\activate.bat"

echo [Pixel] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto FAIL

echo [Pixel] Downloading SigLIP model (first run)...
python -c "from transformers import AutoProcessor, AutoModel; m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m); print('Model loaded:', m)"
if errorlevel 1 goto FAIL

echo [Pixel] Setup complete.
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
echo [Pixel] Uninstall complete.
pause
goto :EOF

:FAIL
echo [Pixel] Error. See output above.
pause
exit /b 1
