import hashlib
import os
import tempfile
from typing import Optional

from PIL import Image, ImageOps


# Глобальный кэш: {cache_key: temp_file_path}
_cache: dict[str, str] = {}

# Обратный индекс: {path: set(cache_key)} — позволяет инвалидировать
# записи по пути независимо от mtime/size (ключ кэша включает их).
_path_keys: dict[str, set[str]] = {}


def _make_cache_key(path: str, mtime: float, size: int) -> str:
    """Создаёт ключ кэша из пути, mtime и размера."""
    raw = f"{path}\x00{mtime}\x00{size}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _remove_key(key: str) -> None:
    """Удаляет запись кэша по ключу и освобождает временный файл."""
    temp_path = _cache.pop(key, None)
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def invalidate(path: str) -> None:
    """Удаляет thumbnail из кэша, если он там есть.

    Ключи кэша — SHA256(path + mtime + size), поэтому инвалидация по пути
    выполняется через обратный индекс _path_keys, а не по префиксу хэша.
    """
    keys = _path_keys.pop(path, None)
    if not keys:
        return
    for key in list(keys):
        _remove_key(key)
        keys.discard(key)


def clear() -> None:
    """Полностью очищает кэш thumbnail."""
    for temp_path in _cache.values():
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    _cache.clear()
    _path_keys.clear()


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
            _cache[cache_key] = temp_path
            # Регистрируем путь в обратном индексе для корректной инвалидации
            _path_keys.setdefault(path, set()).add(cache_key)
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
