#!/bin/bash
cd "$(dirname "$0")"

echo "=== Полное удаление Pixel ==="
echo "Это удалит:"
echo "  - Виртуальное окружение .venv"
echo "  - Кэш модели SigLIP"
echo "  - Базу данных storage/*.db"
echo "  - Маркеры установки"
if [ -d "pixel" ]; then
    echo "  - Папку проекта pixel"
fi
echo ""

read -p "Продолжить? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Отменено."
    exit 0
fi

if [ -d ".venv" ]; then
    echo "Удаляю .venv..."
    rm -rf ".venv"
else
    echo ".venv не найден."
fi

if [ -d "storage" ]; then
    echo "Удаляю базу данных..."
    rm -f storage/*.db storage/*.db-journal storage/*.db-wal storage/*.db-shm 2>/dev/null || true
else
    echo "storage не найден."
fi

if [ -d "pixel" ]; then
    echo "Удаляю папку проекта pixel..."
    rm -rf "pixel"
fi

echo "Удаляю кэш модели Hugging Face..."
rm -rf ~/.cache/huggingface/hub/models--google--siglip-base-patch16-224 2>/dev/null || true
rm -rf ~/.cache/torch/transformers/google--siglip-base-patch16-224 2>/dev/null || true

echo ""
echo "=== Удаление завершено ==="
read -p "Нажмите Enter для выхода..."
