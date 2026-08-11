@echo off
setlocal enabledelayedexpansion

set "REQ_MAJOR=3"
set "REQ_MINOR=12"
set "PYTHON_INSTALL_VER=3.12.9"
set "TEMP_DIR=%TEMP%\pixel_install"
set "REPO_ZIP=%TEMP_DIR%\pixel.zip"
set "REPO_URL=https://github.com/alekcangp/pixel/archive/refs/heads/master.zip"

if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

:: ---------- 0. Скачивание проекта с GitHub, если нет ----------
if not exist "%~dp0pixel\main.py" (
    echo Проект не найден. Скачиваю с GitHub...
    powershell -Command "& {Invoke-WebRequest -Uri '%REPO_URL%' -OutFile '%REPO_ZIP%' -UseBasicParsing}"
    echo Распаковываю...
    powershell -Command "& {Expand-Archive -Path '%REPO_ZIP%' -DestinationPath '%~dp0' -Force}"
    :: GitHub создает папку pixel-master, переименовываем в pixel
    if exist "%~dp0pixel-master" (
        ren "%~dp0pixel-master" pixel
    )
)

:: Переходим в папку проекта
cd /d "%~dp0pixel"

:: ---------- 1. Проверка / установка Python ----------
python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
    echo Найден Python !PY_VER!
    for /f "tokens=1 delims=." %%a in ("!PY_VER!") do set "FOUND_MAJOR=%%a"
    for /f "tokens=2 delims=." %%b in ("!PY_VER!") do set "FOUND_MINOR=%%b"
) else (
    set "FOUND_MAJOR=0"
    set "FOUND_MINOR=0"
)

:: PyTorch пока не поддерживает Python 3.13+, поэтому принудительно ставим 3.12
set "PYTHON_INCOMPATIBLE=0"
if !FOUND_MAJOR! GTR 3 set "PYTHON_INCOMPATIBLE=1"
if !FOUND_MAJOR! EQU 3 if !FOUND_MINOR! GEQ 13 set "PYTHON_INCOMPATIBLE=1"

if !FOUND_MAJOR! LSS %REQ_MAJOR% (
    goto DOWNLOAD_PYTHON
) else if !FOUND_MAJOR! EQU %REQ_MAJOR% (
    if !FOUND_MINOR! LSS %REQ_MINOR% goto DOWNLOAD_PYTHON
)

if !PYTHON_INCOMPATIBLE! EQU 1 (
    echo Python !PY_VER! найден, но torch пока не поддерживает 3.13+. Устанавливаю Python %PYTHON_INSTALL_VER%...
    goto DOWNLOAD_PYTHON
)

goto INSTALL_VC

:DOWNLOAD_PYTHON
echo Python %REQ_MAJOR%.%REQ_MINOR%+ не найден. Скачиваю Python %PYTHON_INSTALL_VER%...

:: Определяем архитектуру
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

:: Скачиваем через PowerShell (curl в Windows иногда вызывает проблемы)
powershell -Command "& {Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_EXE%' -UseBasicParsing}"

echo Устанавливаю Python...
:: /quiet — тихая установка, InstallAllUsers=1 — для всех пользователей, PrependPath=1 — добавить в PATH
start /wait "" "%PYTHON_EXE%" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0

:: Обновляем PATH
call :RefreshEnv

:: Проверяем
python --version >nul 2>&1
if errorlevel 1 (
    echo Не удалось установить Python. Установите вручную.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
echo Установлен Python !PY_VER!

:INSTALL_VC
:: ---------- 2. Visual C++ Redistributable ----------
echo Проверяю Visual C++ Redistributable...
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
echo Visual C++ Redistributable готов.

:VENV_SETUP
:: ---------- 3. venv ----------
set "VENV_NEED_NEW=0"
if not exist ".venv" (
    set "VENV_NEED_NEW=1"
) else (
    :: Если venv создан не под python3.12 — пересоздаём
    if not exist ".venv\Scripts\python.exe" (
        set "VENV_NEED_NEW=1"
    )
)

if !VENV_NEED_NEW! EQU 1 (
    echo Создаю виртуальное окружение...
    python -m venv .venv
) else (
    echo Виртуальное окружение .venv уже существует.
)

call .venv\Scripts\activate.bat

:: ---------- 4. Зависимости ----------
if not exist ".venv\.deps_installed" (
    echo Устанавливаю зависимости Python...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    type nul > .venv\.deps_installed
) else (
    echo Зависимости уже установлены.
)

:: ---------- 5. Модель SigLIP ----------
set "HF_HUB_DISABLE_TELEMETRY=1"
set "TRANSFORMERS_NO_ADVISORY_WARNINGS=1"

if not exist ".venv\.model_downloaded" (
    echo Скачиваю модель SigLIP в кэш...
    python -c "from transformers import AutoProcessor, AutoModel; m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m); print('Модель загружена:', m)"
    type nul > .venv\.model_downloaded
) else (
    echo Модель уже загружена (флаг .venv\.model_downloaded найден).
)

:: ---------- 6. Запуск ----------
echo Запускаю Pixel...
python main.py ui-flet

endlocal
exit /b 0

:RefreshEnv
:: Обновляем PATH из реестра
for /f "skip=2 tokens=*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do (
    set "NEW_PATH=%%a"
)
for /f "tokens=1,* delims=     " %%a in ("!NEW_PATH!") do (
    set "NEW_PATH=%%b"
)
set "PATH=!NEW_PATH!"
exit /b 0
