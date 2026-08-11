# Image Deduplication

Дедупликация, кластеризация и семантический поиск изображений.

**🌐 Язык:** [Русский](README.md) · [English](README.en.md)

## Скриншот

![Image Deduplication UI](screenshots/main.png)

## Основная идея

Все ваши фотографии со всего компьютера — в одном месте. Нажмите одну кнопку — приложение просканирует весь ПК и покажет все изображения в едином представлении: не нужно вручную ходить по папкам. Дальше — автоматический конвейер из нескольких этапов:

1. **Сканирование** — поиск всех изображений по всему ПК (инкрементально, пересчитывается только изменившееся).
2. **Очистка** — исключение одинаковых и визуально похожих дубликатов (pHash + LSH).
3. **Категоризация** — автоматическая разбивка на похожие группы (HDBSCAN + UMAP).
4. **Визуальный отбор** — удобный просмотр и отбор нужных фото в десктопном интерфейсе.
5. **Экспорт** — выгрузка отобранного набора.

Всё работает локально (без загрузки данных в облако) и автоматически. Не требует мощной видеокарты: где есть GPU, используется ускорение (CUDA на NVIDIA, MPS на macOS Silicon), в остальных случаях — CPU. Доступны и десктопный интерфейс на Flet, и CLI.

## Возможности

- Сканирование директорий с инкрементальным обновлением
- Дедупликация по перцептивному хэшу (pHash + LSH)
- Семантический поиск по тексту (SigLIP 2 + FAISS)
- Кластеризация (HDBSCAN + UMAP)
- Desktop UI на Flet

## Требования

- Python 3.12 (рекомендуется)
- pip

## Установка

### Автоматическая установка (рекомендуется)

Самый простой способ — запустить скрипт автоматической установки:

**macOS / Linux:**
```bash
chmod +x install_and_run.command
./install_and_run.command
```

Скрипт сам установит Xcode Command Line Tools, Homebrew, Python 3.12, виртуальное окружение, зависимости и скачает модель SigLIP.

**Windows:**
```bat
install_and_run.bat
```

Скрипт сам установит Python 3.12, Visual C++ Redistributable, виртуальное окружение, зависимости и скачает модель SigLIP.

### Ручная установка

Если автоматическая установка не подходит:

```bash
python -m venv .venv
.venv\Scripts\activate.bat   # Windows (cmd.exe или PowerShell)
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Скачать модель SigLIP (необходима для `ui-flet`, `embed`, `search`):

```bash
python -c "from transformers import AutoProcessor, AutoModel; m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m); print('Model loaded:', m)"
```

## Использование

### Desktop UI

```bash
python main.py ui-flet
```

Интерфейс (см. скриншот выше) доступен после установки зависимостей и загрузки модели SigLIP.

### CLI

```bash
# Сканирование директории
python main.py scan --path "/path/to/folder"

# Полный пересчёт (вместо инкрементального)
python main.py scan --path "/path/to/folder" --full

# Сканирование с фильтром расширений и исключениями
python main.py scan --path "/path/to/folder" --ext "jpg,png,webp" --exclude "tmp,cache"

# Дедупликация
python main.py dedup

# Дедупликация с перемещением дубликатов
python main.py dedup --move "/path/to/duplicates"

# Сканирование + дедупликация за один проход
python main.py run --path "/path/to/folder"

# Очистка базы данных
python main.py clear

# Эмбеддинги (требует PyTorch)
python main.py embed

# Кластеризация
python main.py cluster-hdb

# Семантический поиск (требует PyTorch)
python main.py search "кот на окне"

# Семантический поиск с указанием количества результатов
python main.py search "кот на окне" --top-k 10

# Поиск похожих изображений по pHash
python main.py phash-search "/path/to/image.jpg"

# Поиск похожих изображений по pHash с указанием количества результатов
python main.py phash-search "/path/to/image.jpg" --top-k 10
```

## Примечания

- **Без PyTorch** работают команды `scan`, `dedup`, `run`, `clear`, `cluster-hdb`.
- Команды `embed`, `search` и `phash-search` требуют установленного `torch`.
- По умолчанию используется инкрементальный режим. Добавьте `--full` для полного пересчёта.
- На macOS Silicon автоматически используется MPS, на NVIDIA — CUDA, иначе CPU.
- Размер файла по умолчанию минимум 10 КБ. Используйте `--min-size` для изменения.
