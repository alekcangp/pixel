"""Персистентный кэш миниатюр (WebP BLOB) в SQLite.

Ключевое отличие от старой схемы (temp PNG + in-memory dict):
 * миниатюра генерируется ОДИН раз за время жизни файла (пока не
   изменились mtime/size) и сохраняется в таблицу thumbnails;
 * после перезапуска приложения галерея читает готовые BLOB из БД
   без повторного декодирования оригиналов (0 CPU, единицы мс);
 * формат WebP (q82, method=6): в ~8 раз меньше PNG и быстрее
   передаётся в UI (Flet 0.86.5 поддерживает bytes/src).
"""

import io
import os
from typing import Optional

from PIL import Image, ImageOps

from core import database

# Размер миниатюры, используемый галереей (150×150 квадрат)
THUMB_SIZE = 150
WEBP_QUALITY = 82
WEBP_METHOD = 6


def invalidate(path: str) -> None:
    """Удаляет миниатюру из БД по пути."""
    database.delete_thumbnail(path)


def clear() -> None:
    """Полностью очищает таблицу thumbnails в БД."""
    database.clear_thumbnails()


def get_thumbnail(
    path: str,
    size: int = 150,
    mtime: Optional[float] = None,
    file_size: Optional[int] = None,
) -> Optional[bytes]:
    """Возвращает bytes WebP-миниатюры или None при ошибке.

    Порядок разрешения:
      1. SQLite (таблица thumbnails, с проверкой актуальности mtime/size);
      2. генерация из оригинала (PIL) + сохранение BLOB в БД.
    """
    if not os.path.exists(path):
        return None

    try:
        if mtime is None:
            mtime = os.path.getmtime(path)
        if file_size is None:
            file_size = os.path.getsize(path)

        data = database.load_thumbnail(path, mtime, file_size)
        if data is not None:
            return data

        blob = _generate_thumbnail(path, size)
        if blob is not None and size == THUMB_SIZE:
            database.save_thumbnail(path, mtime, file_size, blob, ext="webp")
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
