"""Персистентный кэш миниатюр (WebP BLOB) в SQLite.

Ключевое отличие от старой схемы (temp PNG + in-memory dict):
  * миниатюра генерируется ОДИН раз за время жизни файла (пока не
    изменились mtime/size) и сохраняется в таблицу thumbnails;
  * после перезапуска приложения галерея читает готовые BLOB из БД
    без повторного декодирования оригиналов (0 CPU, единицы мс);
  * формат WebP (q82, method=6): в ~8 раз меньше PNG и быстрее
    передаётся в UI (Flet 0.86.5 поддерживает bytes/src).
"""

import collections
import io
import os
import threading
from typing import Optional

from PIL import Image, ImageOps

from core import database

# --- Oграниченный LRU в памяти: {key: bytes} -----------------------------
# Хранит небольшой «горячий» набор последних миниатюр, чтобы не ходить
# в SQLite на каждый поворот колеса в галерее.
_LRU_MAX = 2000
_lru: "collections.OrderedDict[str, bytes]" = collections.OrderedDict()
_lru_lock = threading.Lock()

# Размер миниатюры, используемый галереей (150×150 квадрат)
THUMB_SIZE = 150
WEBP_QUALITY = 82
WEBP_METHOD = 6


def _make_cache_key(path: str, mtime: float, size: int) -> str:
    """Ключ LRU: путь + актуальные mtime/size (совпадает с семантикой БД)."""
    return f"{path}\x00{mtime}\x00{size}"


def _cache_get(key: str) -> Optional[bytes]:
    with _lru_lock:
        data = _lru.get(key)
        if data is not None:
            _lru.move_to_end(key)
            return data
        return None


def _cache_put(key: str, data: bytes) -> None:
    with _lru_lock:
        _lru[key] = data
        _lru.move_to_end(key)
        while len(_lru) > _LRU_MAX:
            _lru.popitem(last=False)


def invalidate(path: str) -> None:
    """Удаляет миниатюру из БД и памяти (по пути).

    БД-запись ключуется путём + mtime/size, поэтому при чтении изменённый
    файл всё равно получит перегенерацию; эта функция нужна для принудительной
    чистки (например, при удалении файла).
    """
    prefix = path + "\x00"
    with _lru_lock:
        for key in [k for k in _lru if k.startswith(prefix)]:
            _lru.pop(key, None)
    database.delete_thumbnail(path)


def clear() -> None:
    """Полностью очищает кэш миниатюр: память + таблицу thumbnails в БД."""
    with _lru_lock:
        _lru.clear()
    database.clear_thumbnails()


def get_thumbnail(
    path: str,
    size: int = 150,
    mtime: Optional[float] = None,
    file_size: Optional[int] = None,
) -> Optional[bytes]:
    """Возвращает bytes WebP-миниатюры или None при ошибке.

    Порядок разрешения:
      1. LRU в памяти;
      2. SQLite (таблица thumbnails, с проверкой актуальности mtime/size);
      3. генерация из оригинала (PIL) + сохранение BLOB в БД.

    Result кэшируется навсегда (пока файл не изменён), так что повторное
    открытие галереи/вкладок после перезапуска работает без регенерации.
    """
    if not os.path.exists(path):
        return None

    try:
        if mtime is None:
            mtime = os.path.getmtime(path)
        if file_size is None:
            file_size = os.path.getsize(path)

        cache_key = _make_cache_key(path, mtime, file_size)
        data = _cache_get(cache_key)
        if data is not None:
            return data

        # Персистентный слой: читаем из БД (актуальность проверяет mtime/size).
        data = database.load_thumbnail(path, mtime, file_size)
        if data is not None:
            _cache_put(cache_key, data)
            return data

        # Промах — генерируем. В БД сохраняем только «каноничный» размер 150,
        # остальные размеры возвращаем без кеширования.
        blob = _generate_thumbnail(path, size)
        if blob is not None:
            if size == THUMB_SIZE:
                database.save_thumbnail(path, mtime, file_size, blob, ext="webp")
            _cache_put(cache_key, blob)
        return blob
    except Exception:
        return None


def get_thumbnails_for_paths(paths, mtime_sizes):
    """Батч-чтение миниатюр из БД одним запросом для списка путей.

    Args:
        paths: list[str]
        mtime_sizes: dict {path: (mtime, size)} — актуальные метаданные.

    Returns:
        dict {path: bytes} — только валидные (сохранённые ранее) записи.
        Отсутствующие миниатюры нужно запрашивать по одной через
        get_thumbnail (генерация с сохранением в БД).
    """
    return database.load_thumbnails_for_paths(paths, mtime_sizes)


def _generate_thumbnail(path: str, size: int) -> Optional[bytes]:
    """Генерирует квадратную WebP-миниатюру из оригинального файла и возвращает bytes."""
    try:
        with Image.open(path) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            try:
                img = img.convert("RGB")
            except Exception:
                img = img.convert("RGBA").convert("RGB")

            img.thumbnail((size, size), Image.Resampling.LANCZOS)

            # Добавляем чёрные рамки до квадрата (как было в PNG-версии)
            thumb_w, thumb_h = img.size
            side = max(thumb_w, thumb_h)
            canvas = Image.new("RGB", (side, side), (0, 0, 0))
            x = (side - thumb_w) // 2
            y = (side - thumb_h) // 2
            canvas.paste(img, (x, y))

            buf = io.BytesIO()
            canvas.save(buf, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
            return buf.getvalue()
    except Exception:
        return None
