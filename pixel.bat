@echo off
rem ============================================================
rem  Pixel - единый скрипт установки/запуска/удаления (Windows)
rem
rem  Использование:
rem    pixel.bat            установить (если нужно) + запустить
rem    pixel.bat install    только установка (venv + зависимости + модель)
rem    pixel.bat run        только запуск
rem    pixel.bat uninstall  удаление (venv, БД, кэш модели)
rem ============================================================
setlocal
cd /d "%~dp0"

if /i "%~1"=="install"   goto INSTALL
if /i "%~1"=="run"       goto RUN
if /i "%~1"=="uninstall" goto UNINSTALL

rem Без аргумента: установка (если venv ещё нет) + запуск
if not exist ".venv\Scripts\python.exe" goto INSTALL
goto RUN

:INSTALL
echo [Pixel] Настраиваю виртуальное окружение...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto FAIL
)
call ".venv\Scripts\activate.bat"

echo [Pixel] Устанавливаю зависимости...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto FAIL

echo [Pixel] Скачиваю модель SigLIP (первый запуск)...
python -c "from transformers import AutoProcessor, AutoModel; m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m); print('Модель загружена', m)"
if errorlevel 1 goto FAIL

echo [Pixel] Установка завершена.
goto :EOF

:RUN
if not exist ".venv\Scripts\python.exe" (
    echo [Pixel] Окружение не найдено. Сначала запустите: pixel.bat install
    goto FAIL
)
echo [Pixel] Запускаю Pixel...
".venv\Scripts\python.exe" main.py ui-flet
goto :EOF

:UNINSTALL
echo [Pixel] Удаляю виртуальное окружение...
if exist ".venv" rmdir /s /q ".venv"
echo [Pixel] Удаляю базу данных storage\*.db...
if exist "storage" del /f /q "storage\*.db" "storage\*.db-journal" "storage\*.db-wal" "storage\*.db-shm" 2>nul
echo [Pixel] Удаляю кэш модели Hugging Face...
if exist "%USERPROFILE%\.cache\huggingface\hub\models--google--siglip-base-patch16-224" rmdir /s /q "%USERPROFILE%\.cache\huggingface\hub\models--google--siglip-base-patch16-224"
if exist "%USERPROFILE%\.cache\torch\transformers\google--siglip-base-patch16-224" rmdir /s /q "%USERPROFILE%\.cache\torch\transformers\google--siglip-base-patch16-224"
echo [Pixel] Удаление завершено.
goto :EOF

:FAIL
echo [Pixel] Ошибка. Смотрите вывод выше.
exit /b 1
