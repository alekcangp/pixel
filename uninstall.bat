@echo off
setlocal enabledelayedexpansion

echo === Полное удаление Pixel ===
echo Это удалит:
echo   - Виртуальное окружение .venv
echo   - Базу данных storage\*.db
echo   - Маркеры установки
echo   - Кэш модели Hugging Face
if exist "%~dp0pixel" (
    echo   - Папку проекта pixel
)
echo.

set /p CONFIRM="Продолжить? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo Отменено.
    pause
    exit /b 0
)

if exist "%~dp0pixel\.venv" (
    echo Удаляю .venv...
    rmdir /s /q "%~dp0pixel\.venv"
) else (
    echo .venv не найден.
)

if exist "%~dp0pixel\storage" (
    echo Удаляю базу данных...
    del /f /q "%~dp0pixel\storage\*.db" "%~dp0pixel\storage\*.db-journal" "%~dp0pixel\storage\*.db-wal" "%~dp0pixel\storage\*.db-shm" 2>nul
) else (
    echo storage не найден.
)

if exist "%~dp0pixel" (
    echo Удаляю папку проекта pixel...
    rmdir /s /q "%~dp0pixel"
)

echo Удаляю кэш модели Hugging Face...
if exist "%USERPROFILE%\.cache\huggingface\hub\models--google--siglip-base-patch16-224" (
    rmdir /s /q "%USERPROFILE%\.cache\huggingface\hub\models--google--siglip-base-patch16-224"
)
if exist "%USERPROFILE%\.cache\torch\transformers\google--siglip-base-patch16-224" (
    rmdir /s /q "%USERPROFILE%\.cache\torch\transformers\google--siglip-base-patch16-224"
)

echo.
echo === Удаление завершено ===
pause
