"""Кэш миниатюр для галереи.

Хранит сгенерированные thumbnail в памяти, инвалидирует по ключу:
  path + mtime + size

Это позволяет:
- не пересоздавать thumbnail при каждом rerun UI;
- корректно инвалидировать кэш при изменении файла;
- не зависеть от путей как единственного ключа.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps


# Глобальный кэш: {cache_key: temp_file_path}
_cache: dict[str, str] = {}


def _make_cache_key(path: str, mtime: float, size: int) -> str:
    """Создаёт ключ кэша из пути, mtime и размера."""
    raw = f"{path}\x00{mtime}\x00{size}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def invalidate(path: str) -> None:
    """Удаляет thumbnail из кэша, если он там есть."""
    prefix = hashlib.sha256(path.encode("utf-8", errors="replace")).hexdigest()
    for key in list(_cache.keys()):
        if key.startswith(prefix):
            temp_path = _cache.pop(key, None)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


def clear() -> None:
    """Полностью очищает кэш thumbnail."""
    for temp_path in _cache.values():
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    _cache.clear()


def get_thumbnail(
    path: str,
    size: int = 150,
    mtime: Optional[float] = None,
    file_size: Optional[int] = None,
) -> Optional[str]:
    """Возвращает путь к временному PNG-файлу thumbnail или None при ошибке.

    Args:
        path: путь к файлу.
        size: целевой размер квадратной миниатюры в пикселях.
        mtime: время модификации файла (если None — берётся из файла).
        file_size: размер файла в байтах (если None — берётся из файла).

    Returns:
        str с путём к PNG-файлу или None.
    """
    if not os.path.exists(path):
        return None

    try:
        if mtime is None:
            mtime = os.path.getmtime(path)
        if file_size is None:
            file_size = os.path.getsize(path)

        cache_key = _make_cache_key(path, mtime, file_size)
        cached_path = _cache.get(cache_key)
        if cached_path is not None and os.path.exists(cached_path):
            return cached_path

        temp_path = _generate_thumbnail(path, size)
        if temp_path is not None:
            # Удаляем старый файл кэша для этого пути, если есть
            old_key = _make_cache_key(path, mtime, file_size)
            old_path = _cache.pop(old_key, None)
            if old_path and os.path.exists(old_path):
                try:
                    os.unlink(old_path)
                except Exception:
                    pass
            _cache[cache_key] = temp_path
        return temp_path
    except Exception:
        return None


def _generate_thumbnail(path: str, size: int) -> Optional[str]:
    """Генерирует thumbnail из изображения и сохраняет во временный файл."""
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

            # Добавляем чёрные рамки до квадрата
            thumb_w, thumb_h = img.size
            side = max(thumb_w, thumb_h)
            canvas = Image.new("RGB", (side, side), (0, 0, 0))
            x = (side - thumb_w) // 2
            y = (side - thumb_h) // 2
            canvas.paste(img, (x, y))

            # Сохраняем во временный файл
            fd, temp_path = tempfile.mkstemp(suffix=".png")
            try:
                os.close(fd)
                canvas.save(temp_path, format="PNG", optimize=True)
                return temp_path
            except Exception:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                return None
    except Exception:
        return None
