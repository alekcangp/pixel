@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "REQ_MAJOR=3"
set "REQ_MINOR=12"
set "PYTHON_INSTALL_VER=3.12.9"
set "TEMP_DIR=%TEMP%\pixel_install"
set "REPO_ZIP=%TEMP_DIR%\pixel.zip"
set "REPO_URL=https://github.com/alekcangp/pixel/archive/refs/heads/master.zip"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

:: ---------- 0. Download project from GitHub if needed ----------
if not exist "%~dp0pixel\main.py" (
    echo Project not found. Downloading from GitHub...
    powershell -Command "& {Invoke-WebRequest -Uri '%REPO_URL%' -OutFile '%REPO_ZIP%' -UseBasicParsing}"
    echo Extracting...
    powershell -Command "& {Expand-Archive -Path '%REPO_ZIP%' -DestinationPath '%~dp0' -Force}"
    :: GitHub creates pixel-master folder, rename to pixel
    if exist "%~dp0pixel-master" (
        ren "%~dp0pixel-master" pixel
    )
)

:: Change to project folder
cd /d "%~dp0pixel"

:: ---------- 1. Check / install Python ----------
python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
    echo Found Python !PY_VER!
    for /f "tokens=1 delims=." %%a in ("!PY_VER!") do set "FOUND_MAJOR=%%a"
    for /f "tokens=2 delims=." %%b in ("!PY_VER!") do set "FOUND_MINOR=%%b"
) else (
    set "FOUND_MAJOR=0"
    set "FOUND_MINOR=0"
)

:: PyTorch does not support Python 3.13+, force install 3.12
set "PYTHON_INCOMPATIBLE=0"
if !FOUND_MAJOR! GTR 3 set "PYTHON_INCOMPATIBLE=1"
if !FOUND_MAJOR! EQU 3 if !FOUND_MINOR! GEQ 13 set "PYTHON_INCOMPATIBLE=1"

if !FOUND_MAJOR! LSS %REQ_MAJOR% (
    goto DOWNLOAD_PYTHON
) else if !FOUND_MAJOR! EQU %REQ_MAJOR% (
    if !FOUND_MINOR! LSS %REQ_MINOR% goto DOWNLOAD_PYTHON
)

if !PYTHON_INCOMPATIBLE! EQU 1 (
echo Python !PY_VER! found, but torch does not support 3.13+ yet. Installing Python %PYTHON_INSTALL_VER%...
    goto DOWNLOAD_PYTHON
)

goto INSTALL_VC

:DOWNLOAD_PYTHON
echo Python %REQ_MAJOR%.%REQ_MINOR%+ not found. Downloading Python %PYTHON_INSTALL_VER%...

:: Detect architecture
set "ARCH=AMD64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=ARM64"
if /i "%PROCESSOR_ARCHITECTURE%"=="x86" set "ARCH=win32"

set "PYTHON_EXE=%TEMP_DIR%\python-%PYTHON_INSTALL_VER%-%ARCH%.exe"
if "%ARCH%"=="AMD64" (
    set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_INSTALL_VER%/python-%PYTHON_INSTALL_VER%-amd64.exe"
) else if "%ARCH%"=="win32" (
    set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_INSTALL_VER%/python-%PYTHON_INSTALL_VER%.exe"
) else (
    set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_INSTALL_VER%/python-%PYTHON_INSTALL_VER%-arm64.exe"
)

:: Download via PowerShell (curl problematic on Windows)
powershell -Command "& {Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_EXE%' -UseBasicParsing}"

echo Installing Python...
:: /quiet silent, InstallAllUsers=1 all users, PrependPath=1 add to PATH
start /wait "" "%PYTHON_EXE%" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0

:: Refresh PATH
call :RefreshEnv

:: Verify
python --version >nul 2>&1
if errorlevel 1 (
    echo Failed to install Python. Install it manually.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
echo Installed Python !PY_VER!

:INSTALL_VC
:: ---------- 2. Visual C++ Redistributable ----------
echo Checking Visual C++ Redistributable...
set "VC_EXE=%TEMP_DIR%\vc_redist.exe"
if /i "%ARCH%"=="ARM64" (
    set "VC_URL=https://aka.ms/vs/17/release/vc_redist.arm64.exe"
) else if /i "%ARCH%"=="win32" (
    set "VC_URL=https://aka.ms/vs/17/release/vc_redist.x86.exe"
) else (
    set "VC_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
)

powershell -Command "& {Invoke-WebRequest -Uri '%VC_URL%' -OutFile '%VC_EXE%' -UseBasicParsing}"
start /wait "" "%VC_EXE%" /quiet /norestart
echo Visual C++ Redistributable ready.

:VENV_SETUP
:: ---------- 3. venv ----------
set "VENV_NEED_NEW=0"
if not exist ".venv" (
    set "VENV_NEED_NEW=1"
) else (
    :: Recreate if venv was built for a python other than 3.12
    if not exist ".venv\Scripts\python.exe" (
        set "VENV_NEED_NEW=1"
    )
)

if !VENV_NEED_NEW! EQU 1 (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment .venv already exists.
)

call .venv\Scripts\activate.bat

:: ---------- 4. Dependencies ----------
if not exist ".venv\.deps_installed" (
    echo Installing Python dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    type nul > .venv\.deps_installed
) else (
    echo Dependencies already installed.
)

:: ---------- 5. SigLIP model ----------
set "HF_HUB_DISABLE_TELEMETRY=1"
set "TRANSFORMERS_NO_ADVISORY_WARNINGS=1"

if not exist ".venv\.model_downloaded" (
    echo Downloading SigLIP model to cache...
    python -c "from transformers import AutoProcessor, AutoModel; m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m); print('Model loaded:', m)"
    type nul > .venv\.model_downloaded
) else (
    echo Model already downloaded (flag .venv\.model_downloaded found).
)

:: ---------- 6. Run ----------
echo Starting Pixel...
python main.py ui-flet

endlocal
exit /b 0

:RefreshEnv
:: Refresh PATH from registry
for /f "skip=2 tokens=*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do (
    set "NEW_PATH=%%a"
)
for /f "tokens=1,* delims=     " %%a in ("!NEW_PATH!") do (
    set "NEW_PATH=%%b"
)
set "PATH=!NEW_PATH!"
exit /b 0
