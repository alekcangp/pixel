"""Локализация интерфейса (русский / английский).

По умолчанию язык берётся из config.APP_LANGUAGE ("auto"). При значении "auto"
язык определяется по локали ОС (префикс "ru" → русский, иначе английский).
Выбор пользователя сохраняется в storage/lang.txt и при следующем запуске
перекрывает значение по умолчанию.
"""

import locale
import os

import config

STRINGS = {
    "ru": {
        "app.title": "Pixel",
        "header.title": "Pixel",
        "lang": "Язык",
        "lang.ru": "Русский",
        "lang.en": "English",
        "tab.search": "Поиск",
        "tab.export": "Экспорт",
        "loading": "Загрузка...",
        "stat.title": "Статистика",
        "stat.total": "Всего",
        "stat.dupes": "Дубли",
        "stat.unique": "Уникальных",
        "stat.selected": "Выбрано",
        "categories.title": "Категории ({count})",
        "categories.empty": "Категории ещё не созданы. Запустите полный цикл.",
        "model.loading": "Загрузка модели...",
        "model.loaded": "✅ Модель загружена",
        "model.error": "⚠️ Ошибка: {error}",
        "scan.path.label": "Путь для сканирования",
        "scan.path.hint": "Выберите папку для сканирования",
        "scan.browse": "Выбрать папку",
        "scan.all_disks": "Все диски",
        "scan.button.scan": "Сканировать",
        "scan.button.start": "Запустить сканирование",
        "scan.button.stop": "Стоп",
        "scan.button.reset": "Сброс",
        "scan.no_files": "Файлы не найдены",
        "scan.no_disks": "Не найдено доступных дисков",
        "scan.bad_path": "Путь не существует: {path}",
        "scan.stopped": "Сканирование остановлено",
        "stage.scan": "Сканирование",
        "stage.dedup": "Дедупликация",
        "stage.embed": "Эмбеддинги",
        "stage.cluster": "Кластеризация",
        "stage.thumbs": "Миниатюры",
        "stage.export": "Экспорт",
        "progress.prepare": "Подготовка...",
        "progress.scan": "Сканирование ({cur}/{total}): {path}",
        "progress.dedup": "Дедупликация...",
        "progress.embed": "Эмбеддинги...",
        "progress.cluster": "Кластеризация...",
        "progress.export.prepare": "Экспорт: подготовка...",
        "progress.export.step": "Экспорт: {cur}/{total} ({percent}%)",
        "ready": "Готово!",
        "error": "Ошибка: {error}",
        "reset.title": "Подтверждение сброса",
        "reset.prompt": "Это удалит всю базу данных и кэш. Продолжить?",
        "reset.confirm": "Сбросить",
        "reset.done": "Сброс выполнен. База и кэш очищены.",
        "cancel": "Отмена",
        "export.selected_files": "Выбрано файлов",
        "export.total_size": "Общий размер",
        "export.folder": "Папка назначения",
        "export.default_folder": "Фотоальбом",
        "export.button": "Экспортировать выбранные",
        "export.button.stop": "Стоп",
        "export.stopped": "Экспорт остановлен",
        "export.none": "Ничего не выбрано. Выберите изображения в галерее.",
        "export.none_selected": "Нет выбранных файлов для экспорта",
        "export.copied": "Скопировано {count} файлов в {dest}",
        "select_all.toggle": "Выбрать/Снять все",
        "selected.count": "Выбрано: {count}",
        "selected.files": "Выбрано: {count} файлов",
        "search.label": "Поиск по описанию",
        "search.hint": "например: кот на окне",
        "search.button": "Найти",
        "search.empty_query": "Введите запрос",
        "search.running": "Поиск...",
        "search.results": "Найдено: {count}",
        "search.none": "Ничего не найдено",
        "search.error": "Ошибка поиска: {error}",
        "gallery.empty": "Нет изображений",
        "preview.open_location": "Открыть расположение файла",
        "context.zoom_in": "Увеличить",
        "context.open_location": "Расположение",
        "pick.scan": "Выберите папку для сканирования",
        "pick.export": "Выберите папку назначения",
        "unit.B": "Б",
        "unit.KB": "КБ",
        "unit.MB": "МБ",
        "unit.GB": "ГБ",
    },
    "en": {
        "app.title": "Pixel",
        "header.title": "Pixel",
        "lang": "Language",
        "lang.ru": "Russian",
        "lang.en": "English",
        "tab.search": "Search",
        "tab.export": "Export",
        "loading": "Loading...",
        "stat.title": "Statistics",
        "stat.total": "Total",
        "stat.dupes": "Duplicates",
        "stat.unique": "Unique",
        "stat.selected": "Selected",
        "categories.title": "Categories ({count})",
        "categories.empty": "Categories have not been created yet. Run the full pipeline.",
        "model.loading": "Loading model...",
        "model.loaded": "✅ Model loaded",
        "model.error": "⚠️ Error: {error}",
        "scan.path.label": "Path to scan",
        "scan.path.hint": "Select a folder to scan",
        "scan.browse": "Choose folder",
        "scan.all_disks": "All disks",
        "scan.button.scan": "Scan",
        "scan.button.start": "Start scan",
        "scan.button.stop": "Stop scan",
        "scan.button.reset": "Reset",
        "scan.no_files": "No files found",
        "scan.no_disks": "No available disks found",
        "scan.bad_path": "Path does not exist: {path}",
        "scan.stopped": "Scanning stopped",
        "stage.scan": "Scanning",
        "stage.dedup": "Deduplication",
        "stage.embed": "Embeddings",
        "stage.cluster": "Clustering",
        "stage.thumbs": "Thumbnails",
        "stage.export": "Export",
        "progress.prepare": "Preparing...",
        "progress.scan": "Scanning ({cur}/{total}): {path}",
        "progress.dedup": "Deduplicating...",
        "progress.embed": "Embeddings...",
        "progress.cluster": "Clustering...",
        "progress.export.prepare": "Export: preparing...",
        "progress.export.step": "Export: {cur}/{total} ({percent}%)",
        "ready": "Done!",
        "error": "Error: {error}",
        "reset.title": "Confirm reset",
        "reset.prompt": "This will delete the entire database and cache. Continue?",
        "reset.confirm": "Reset",
        "reset.done": "Reset complete. Database and cache cleared.",
        "cancel": "Cancel",
        "export.selected_files": "Files selected",
        "export.total_size": "Total size",
        "export.folder": "Destination folder",
        "export.default_folder": "Photo Album",
        "export.button": "Export selected",
        "export.button.stop": "Stop",
        "export.stopped": "Export stopped",
        "export.none": "Nothing selected. Select images in the gallery.",
        "export.none_selected": "No files selected for export",
        "export.copied": "Copied {count} files to {dest}",
        "select_all.toggle": "Select / Deselect all",
        "selected.count": "Selected: {count}",
        "selected.files": "Selected: {count} files",
        "search.label": "Search by description",
        "search.hint": "e.g. cat on the window",
        "search.button": "Find",
        "search.empty_query": "Enter a query",
        "search.running": "Searching...",
        "search.results": "Found: {count}",
        "search.none": "Nothing found",
        "search.error": "Search error: {error}",
        "gallery.empty": "No images",
        "preview.open_location": "Open file location",
        "context.zoom_in": "Zoom in",
        "context.open_location": "Location",
        "pick.scan": "Select a folder to scan",
        "pick.export": "Select destination folder",
        "unit.B": "B",
        "unit.KB": "KB",
        "unit.MB": "MB",
        "unit.GB": "GB",
    },
}


def _init_lang() -> str:
    lang = getattr(config, "APP_LANGUAGE", "ru")
    if lang == "auto":
        loc = ""
        try:
            loc = locale.getlocale()[0] or ""
        except Exception:
            pass
        if not loc or loc in ("C", "POSIX"):
            try:
                loc = locale.getdefaultlocale()[0] or ""
            except Exception:
                pass
        if not loc or loc in ("C", "POSIX"):
            loc = (
                os.environ.get("LANG", "")
                or os.environ.get("LC_ALL", "")
                or os.environ.get("LC_MESSAGES", "")
                or ""
            )
        loc = loc.lower()
        if loc.startswith("ru"):
            lang = "ru"
        else:
            lang = "en"
    return lang


LANG = _init_lang()


def tr(key: str, **kw) -> str:
    """Возвращает перевод строки по ключу с подстановкой плейсхолдеров."""
    text = STRINGS[LANG][key]
    if kw:
        return text.format(**kw)
    return text


def set_language(lang: str) -> None:
    """Переключает язык интерфейса."""
    global LANG
    lang = lang if lang in STRINGS else "ru"
    LANG = lang


def is_all_disks(value: str) -> bool:
    """Проверяет, что значение пути соответствует сентинелю «все диски»."""
    v = (value or "").strip().lower()
    if v == "all_disks":
        return True
    return any(STRINGS[lang]["scan.all_disks"].lower() == v for lang in STRINGS)
