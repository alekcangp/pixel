#!/bin/bash
# ============================================================
#  Pixel - единый скрипт установки/запуска/удаления (macOS)
#
#  Использование:
#    ./pixel.command            установить (если нужно) + запустить
#    ./pixel.command install    только установка (venv + зависимости + модель)
#    ./pixel.command run        только запуск
#    ./pixel.command uninstall  удаление (venv, БД, кэш модели)
# ============================================================
cd "$(dirname "$0")"

CMD="${1:-}"

log() { echo -e "\033[1;34m[Pixel]\033[0m $*"; }
fail() { echo -e "\033[1;31m[Pixel] Ошибка:\033[0m $*" >&2; exit 1; }

# Без аргумента: установка (если venv ещё нет) + запуск
if [ -z "$CMD" ]; then
    if [ ! -x ".venv/bin/python3" ]; then
        CMD="install"
    else
        CMD="run"
    fi
fi

case "$CMD" in
    install)
        log "Настраиваю виртуальное окружение..."
        if [ ! -x ".venv/bin/python3" ]; then
            python3 -m venv .venv || fail "не удалось создать venv"
        fi
        source .venv/bin/activate

        log "Устанавливаю зависимости..."
        python -m pip install --upgrade pip
        pip install -r requirements.txt || fail "pip install"

        log "Скачиваю модель SigLIP (первый запуск)..."
        python - <<'PY' || fail "скачивание модели"
from transformers import AutoProcessor, AutoModel
m = "google/siglip-base-patch16-224"
AutoProcessor.from_pretrained(m)
AutoModel.from_pretrained(m)
print("Модель загружена", m)
PY
        log "Установка завершена."
        ;;
    run)
        if [ ! -x ".venv/bin/python3" ]; then
            fail "окружение не найдено. Сначала: ./pixel.command install"
        fi
        log "Запускаю Pixel..."
        ./.venv/bin/python main.py ui-flet
        ;;
    uninstall)
        log "Удаляю виртуальное окружение..."
        rm -rf ".venv"
        log "Удаляю базу данных storage/*.db..."
        rm -f storage/*.db storage/*.db-journal storage/*.db-wal storage/*.db-shm 2>/dev/null || true
        log "Удаляю кэш модели Hugging Face..."
        rm -rf ~/.cache/huggingface/hub/models--google--siglip-base-patch16-224 2>/dev/null || true
        rm -rf ~/.cache/torch/transformers/google--siglip-base-patch16-224 2>/dev/null || true
        log "Удаление завершено."
        ;;
    *)
        fail "неизвестная команда '$CMD'. Используйте: install | run | uninstall"
        ;;
esac
