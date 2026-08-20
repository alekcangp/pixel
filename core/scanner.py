import os
import sys
from collections import Counter

import config
from core import database

# Флаг остановки сканирования (устанавливается из UI)
STOP_REQUESTED = False


def _normalize_extensions(extensions):
    if extensions is None:
        extensions = config.DEFAULT_EXTENSIONS
    return [e.lower().lstrip(".") for e in extensions]


def _normalize_exclude_dirs(exclude_dirs):
    if exclude_dirs is None:
        exclude_dirs = []
    return {d.lower() for d in config.DEFAULT_EXCLUDE_DIRS + list(exclude_dirs)}


def _count_matching_files(path, extensions=None, exclude_dirs=None, min_size=None):
    global STOP_REQUESTED
    ext_set = set(_normalize_extensions(extensions))
    exclude_set = _normalize_exclude_dirs(exclude_dirs)
    if min_size is None:
        min_size = config.MIN_FILE_SIZE

    count = 0
    stack = [path]
    while stack:
        if STOP_REQUESTED:
            return count
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if STOP_REQUESTED:
                        return count
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.lower() in exclude_set:
                                continue
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            ext = entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
                            if ext not in ext_set:
                                continue
                            try:
                                size = entry.stat(follow_symlinks=False).st_size
                            except OSError:
                                size = 0
                            if size < min_size:
                                continue
                            count += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return count


def scan_directory(path, extensions=None, exclude_dirs=None, min_size=None, progress_callback=None, total=None):
    """Рекурсивно сканирует директорию, собирает файлы с указанными расширениями.

    total: необязательная подсказка общего числа файлов (если она уже была заранее
    посчитана вызывающей стороной, повторный проход для подсчёта не выполняется).
    Это позволяет GUI избежать тройного обхода дерева (UI-count → scan count → collect).
    """
    ext_set = set(_normalize_extensions(extensions))
    exclude_set = _normalize_exclude_dirs(exclude_dirs)
    if min_size is None:
        min_size = config.MIN_FILE_SIZE

    if total is None:
        total = _count_matching_files(path, extensions=extensions, exclude_dirs=exclude_dirs, min_size=min_size)
    files = []
    processed = 0
    stack = [path]
    while stack:
        if STOP_REQUESTED:
            return files
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if STOP_REQUESTED:
                        return files
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.lower() in exclude_set:
                                continue
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            ext = entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
                            if ext not in ext_set:
                                continue
                            try:
                                stat = entry.stat(follow_symlinks=False)
                                size = stat.st_size
                                mtime = stat.st_mtime
                            except OSError:
                                size = 0
                                mtime = 0
                            if size < min_size:
                                continue
                            files.append({
                                "path": entry.path,
                                "size": size,
                                "ext": ext,
                                "mtime": mtime,
                            })
                            processed += 1
                            if progress_callback is not None:
                                progress_callback("scan", processed, total, "Сканирование файлов")
                    except OSError:
                        continue
        except OSError:
            continue

    return files


def incremental_scan(path, extensions=None, exclude_dirs=None, min_size=None, progress_callback=None, total=None):
    """Инкрементальное сканирование: возвращает (new_files, changed_files, removed_paths, all_files)."""
    ext_set = set(_normalize_extensions(extensions))
    exclude_set = _normalize_exclude_dirs(exclude_dirs)
    if min_size is None:
        min_size = config.MIN_FILE_SIZE

    existing = database.get_existing_images()
    existing_paths = set(existing.keys())

    # Сканируем текущее состояние
    all_files = scan_directory(path, extensions=extensions, exclude_dirs=exclude_dirs, min_size=min_size, progress_callback=progress_callback, total=total)
    all_paths = {f["path"] for f in all_files}

    new_files = []
    changed_files = []
    removed_paths = []

    # Новые и изменённые
    for f in all_files:
        old = existing.get(f["path"])
        if old is None:
            new_files.append(f)
        else:
            if (f["size"] != old["size"] or 
                f["mtime"] != old["mtime"]):
                changed_files.append(f)

    removed_paths = []
    source_norm = os.path.normcase(path) if sys.platform == "win32" else path
    for p in existing_paths - all_paths:
        if os.path.normcase(existing[p].get("source", "")) == source_norm:
            removed_paths.append(p)

    return new_files, changed_files, removed_paths, all_files


def save_index(files):
    database.save_images(files)


def load_index():
    return database.load_images()


def print_stats(files):
    total = len(files)
    total_size = sum(f["size"] for f in files)
    ext_counter = Counter(f["ext"] for f in files)

    print(f"Всего файлов: {total}")
    print(f"Общий размер: {total_size / (1024**2):.2f} MB")
    print("\nСтатистика по расширениям:")
    for ext, cnt in ext_counter.most_common():
        ext_size = sum(f["size"] for f in files if f["ext"] == ext)
        print(f"  .{ext}: {cnt} файлов, {ext_size / (1024**2):.2f} MB")

    print("\nТоп-10 путей:")
    for f in files[:10]:
        print(f"  {f['path']} ({f['size']} bytes)")


def run(path, extensions, exclude_dirs=None, min_size=None, progress_callback=None, incremental=True, total=None):
    if not os.path.isdir(path):
        print("Путь не найден или не является директорией: %s" % path)
        return None
    if min_size is None:
        min_size = config.MIN_FILE_SIZE
    print(f"Сканирование: {path}")
    print(f"Минимальный размер файла: {min_size} байт ({min_size / 1024:.0f} КБ)")
    if exclude_dirs:
        print(f"Исключаемые директории: {', '.join(exclude_dirs)}")

    if incremental:
        new_files, changed_files, removed_paths, all_files = incremental_scan(
            path, extensions, exclude_dirs, min_size=min_size, progress_callback=progress_callback, total=total
        )
        if not all_files:
            print("Файлы не найдены (проверьте путь и расширения).")
            return []
        
        print(f"\nИнкрементальное обновление:")
        print(f"  Новых файлов: {len(new_files)}")
        print(f"  Изменённых: {len(changed_files)}")
        print(f"  Удалённых: {len(removed_paths)}")
        
        database.update_images_incremental(new_files, changed_files, removed_paths, source=path)
        files = all_files
    else:
        files = scan_directory(path, extensions, exclude_dirs, min_size=min_size, progress_callback=progress_callback)
        if not files:
            print("Файлы не найдены (проверьте путь и расширения).")
            return []
        save_index(files)

    print_stats(files)
    print(f"\nИндекс сохранён в БД: {config.DB_FILE}")
    return files