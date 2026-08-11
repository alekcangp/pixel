@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo === Full Pixel Uninstall ===
echo This will remove:
echo   - Virtual environment .venv
echo   - Database storage\*.db
echo   - Installation markers
echo   - Hugging Face model cache
if exist "%~dp0pixel" (
    echo   - pixel project folder
)
echo.

set /p CONFIRM="Continue? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

if exist "%~dp0pixel\.venv" (
    echo Removing .venv...
    rmdir /s /q "%~dp0pixel\.venv"
) else (
    echo .venv not found.
)

if exist "%~dp0pixel\storage" (
    echo Removing database...
    del /f /q "%~dp0pixel\storage\*.db" "%~dp0pixel\storage\*.db-journal" "%~dp0pixel\storage\*.db-wal" "%~dp0pixel\storage\*.db-shm" 2>nul
) else (
    echo storage not found.
)

if exist "%~dp0pixel" (
    echo Removing pixel project folder...
    rmdir /s /q "%~dp0pixel"
)

echo Removing Hugging Face model cache...
if exist "%USERPROFILE%\.cache\huggingface\hub\models--google--siglip-base-patch16-224" (
    rmdir /s /q "%USERPROFILE%\.cache\huggingface\hub\models--google--siglip-base-patch16-224"
)
if exist "%USERPROFILE%\.cache\torch\transformers\google--siglip-base-patch16-224" (
    rmdir /s /q "%USERPROFILE%\.cache\torch\transformers\google--siglip-base-patch16-224"
)

echo.
echo === Uninstall Complete ===
pause
