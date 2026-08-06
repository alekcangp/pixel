"""Дедупликация изображений по перцептивному хэшу (pHash).

Вся логика дедупликации собрана в одном файле:
  - загрузка изображения (OpenCV);
  - вычисление pHash (64-битный int);
  - LSH-индекс для быстрого поиска похожих хэшей;
  - Union-Find для кластеризации;
  - запуск дедупликации и печать статистики.
"""

import os
import sys
import shutil
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

warnings.filterwarnings('ignore')

import config
from core import database
from core.scanner import load_index


# ============================================================
# Загрузка изображения
# ============================================================

def _suppress_cv_stderr():
    """Временно подавляет stderr OpenCV (сообщения об анимированных WebP/GIF)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    orig_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    return orig_stderr_fd, devnull_fd


def _restore_cv_stderr(orig_stderr_fd, devnull_fd):
    """Восстанавливает stderr после _suppress_cv_stderr."""
    os.dup2(orig_stderr_fd, 2)
    os.close(orig_stderr_fd)
    os.close(devnull_fd)


def load_image_cv(path):
    """Загружает изображение через OpenCV (BGR). Возвращает None при ошибке.

    Использует np.fromfile + cv2.imdecode вместо cv2.imread, чтобы корректно
    обрабатывать пути с кириллицей/Unicode на Windows.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        orig_fd, null_fd = _suppress_cv_stderr()
        try:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        finally:
            _restore_cv_stderr(orig_fd, null_fd)
        return img
    except Exception:
        return None


# ============================================================
# pHash
# ============================================================

def _bits_to_int(bits):
    """Преобразует байты OpenCV img_hash в 64-битный int.

    cv2.img_hash.PHash возвращает 8 байтов (64 бита). Распаковываем
    каждый байт в 8 битов и собираем целое число.
    """
    arr = np.asarray(bits).ravel()
    if arr.size == 0:
        return 0

    # Если вход уже выглядит как список битов 0/1 — собираем напрямую.
    if np.all((arr == 0) | (arr == 1)):
        h = 0
        for bit in arr:
            h = (h << 1) | int(bit > 0)
        return int(h)

    # Иначе трактуем значения как байты 0..255 и распаковываем их в биты.
    bytes_arr = arr.astype(np.uint8)
    packed_bits = np.unpackbits(bytes_arr, bitorder="big")
    h = 0
    for bit in packed_bits[:64]:
        h = (h << 1) | int(bit)
    return int(h)


def compute_phash(path):
    """pHash (Perceptual Hash) через OpenCV. Возвращает 64-битный int или None."""
    img = load_image_cv(path)
    if img is None:
        return None
    try:
        hasher = cv2.img_hash.PHash_create()
        res = hasher.compute(img)
        return _bits_to_int(res[0].flatten()[:64])
    except Exception:
        return None


# ============================================================
# Union-Find
# ============================================================

class UnionFind:
    """Union-Find с оптимизациями path compression и union by rank."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_clusters(self):
        clusters = {}
        for x in self.parent:
            root = self.find(x)
            clusters.setdefault(root, []).append(x)
        return clusters


# ============================================================
# LSH-индекс для 64-битных хэшей
# ============================================================

class LSHIndex:
    """LSH-индекс для 64-битных хэшей с поддержкой кластеризации и поиска.

    Разбивает 64-битный хэш на B подблоков и строит инвертированный индекс
    (подблок -> индексы). Если два хэша отличаются не более чем на `threshold`
    бит, то минимум B - threshold подблоков совпадают полностью.
    """

    def __init__(self, int_hashes, threshold):
        """
        int_hashes: список (id, hash) или список hash.
        threshold:  максимальное Hamming distance.
        """
        self._n = len(int_hashes)
        self._threshold = threshold

        if self._n > 0 and not isinstance(int_hashes[0], (tuple, list)):
            self._items = [(i, h) for i, h in enumerate(int_hashes)]
        else:
            self._items = list(int_hashes)

        self._hashes = [h for _, h in self._items]

        B = max(8, threshold + 1)
        B = min(B, 64)
        self._B = B

        base = 64 // B
        extra = 64 % B
        self._block_sizes = [base + (1 if i < extra else 0) for i in range(B)]

        self._inverted = defaultdict(list)
        for i, (_, h) in enumerate(self._items):
            offset = 0
            for b in range(B):
                size = self._block_sizes[b]
                shift = 64 - offset - size
                block_val = (h >> shift) & ((1 << size) - 1)
                self._inverted[(b, block_val)].append(i)
                offset += size

    def __len__(self):
        return self._n

    def _block_values(self, h):
        vals = []
        offset = 0
        for b in range(self._B):
            size = self._block_sizes[b]
            shift = 64 - offset - size
            block_val = (h >> shift) & ((1 << size) - 1)
            vals.append((b, block_val))
            offset += size
        return vals

    def _candidate_indices(self, h):
        candidates = set()
        for b, block_val in self._block_values(h):
            for j in self._inverted.get((b, block_val), []):
                candidates.add(j)
        return candidates

    def radius_search(self, query_hash, radius=None):
        """Возвращает список (dist, idx) для всех хэшей в пределах radius."""
        if radius is None:
            radius = self._threshold
        if self._n == 0:
            return []

        candidates = self._candidate_indices(query_hash)
        if not candidates:
            return []

        cand_list = sorted(candidates)
        cand_hashes = np.array([self._hashes[j] for j in cand_list], dtype=np.uint64)
        h_arr = np.full(len(cand_list), query_hash, dtype=np.uint64)
        xor = np.bitwise_xor(h_arr, cand_hashes)
        xor_bytes = xor.view(np.uint8).reshape(-1, 8)
        dists = np.unpackbits(xor_bytes, axis=1).sum(axis=1)

        results = []
        for j, d in zip(cand_list, dists):
            if d <= radius:
                results.append((int(d), j))
        results.sort(key=lambda x: x[0])
        return results

    def query(self, query_hash, k=1, upper_bound=float("inf")):
        """Возвращает k ближайших результатов в пределах upper_bound."""
        results = self.radius_search(query_hash, radius=upper_bound)
        return results[:k]

    def find_clusters(self, progress_callback=None):
        """Кластеризует все хэши через Union-Find.

        Возвращает список групп: [[id, ...], ...] (только группы размером > 1).
        """
        uf = UnionFind()
        n = self._n
        if n == 0:
            return []

        # Оптимизация: banding-подход
        # Для каждого подблока строим группы элементов с одинаковым значением
        # и объединяем только внутри групп.
        block_groups = defaultdict(list)
        for i, (_, h) in enumerate(self._items):
            offset = 0
            for b in range(self._B):
                size = self._block_sizes[b]
                shift = 64 - offset - size
                block_val = (h >> shift) & ((1 << size) - 1)
                block_groups[(b, block_val)].append(i)
                offset += size

        # Для каждого элемента собираем кандидатов по совпадающим подблокам
        # и объединяем через Union-Find
        processed = 0
        for i, (_, h) in enumerate(self._items):
            # Собираем кандидатов по всем подблокам элемента i
            candidates = set()
            offset = 0
            for b in range(self._B):
                size = self._block_sizes[b]
                shift = 64 - offset - size
                block_val = (h >> shift) & ((1 << size) - 1)
                candidates.update(block_groups.get((b, block_val), []))
                offset += size

            if not candidates:
                processed += 1
                continue

            # Векторизованное вычисление Hamming distance
            cand_list = sorted(candidates)
            cand_hashes = np.array([self._hashes[j] for j in cand_list], dtype=np.uint64)
            h_arr = np.full(len(cand_list), h, dtype=np.uint64)
            xor = np.bitwise_xor(h_arr, cand_hashes)
            xor_bytes = xor.view(np.uint8).reshape(-1, 8)
            dists = np.unpackbits(xor_bytes, axis=1).sum(axis=1)

            for j, d in zip(cand_list, dists):
                if d <= self._threshold:
                    uf.union(i, j)

            processed += 1
            if progress_callback is not None:
                progress_callback("dedup", processed, n, "Кластеризация pHash")
            if processed % 100 == 0 or processed == n:
                _progress(processed, n, "Кластеризация")

        clusters = uf.get_clusters()
        groups = []
        for root, indices in clusters.items():
            if len(indices) > 1:
                group = [self._items[idx][0] for idx in indices]
                groups.append(group)
        return groups


# ============================================================
# Дедупликация
# ============================================================

def _progress(current, total, label="Обработка"):
    pct = current * 100 // total if total else 100
    sys.stdout.write("\r%s: %d/%d (%d%%)" % (label, current, total, pct))
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _process_single(path):
    if not os.path.exists(path):
        return None, "not_found"
    h = compute_phash(path)
    if h is None:
        return None, "corrupt"
    return (path, h), "ok"


def find_similar_images(files, max_workers=4, progress_callback=None):
    paths_to_process = [f["path"] for f in files]
    total = len(paths_to_process)

    if total == 0:
        return [], [], {}, {}

    threshold = config.PHASH_THRESHOLD

    hashes = []
    hash_map = {}  # {path: phash_int} — для сохранения в БД
    failed = 0
    failed_paths = []
    failed_reasons = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single, path): path for path in paths_to_process}

        completed = 0
        for future in as_completed(futures):
            path = futures[future]
            result, reason = future.result()
            if result is not None:
                p, h = result
                hashes.append(result)
                hash_map[p] = h
            else:
                failed += 1
                failed_paths.append(path)
                failed_reasons[path] = reason
            completed += 1
            if progress_callback is not None:
                progress_callback("dedup", completed, total, "Поиск похожих изображений")
            if completed % 50 == 0 or completed == total:
                _progress(completed, total, "Похожие изображения (pHash)")

    n = len(hashes)
    if n == 0:
        return [], failed_paths, failed_reasons, hash_map

    print("\nКластеризация %d хэшей..." % n)
    index = LSHIndex(hashes, threshold)
    groups = index.find_clusters(progress_callback=progress_callback)

    return groups, failed_paths, failed_reasons, hash_map


def _pick_canonical(paths):
    """Выбирает «эталонный» файл группы: с самым ранним mtime (оригинал).

    Оригинал обычно старше дубликатов, поэтому выбираем минимальный mtime.
    При равенстве — наибольший размер файла. Не требует декодирования
    изображений, что в десятки раз быстрее предыдущего варианта с
    разрешением через OpenCV.
    """
    return min(
        paths,
        key=lambda p: (os.path.getmtime(p) if os.path.exists(p) else float('inf'),
                       -(os.path.getsize(p) if os.path.exists(p) else 0)),
    )


def _safe_move(src, dst_dir):
    base = os.path.basename(src)
    dst = os.path.join(dst_dir, base)
    if os.path.exists(dst):
        root, ext = os.path.splitext(base)
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(dst_dir, "%s_%d%s" % (root, i, ext))
            i += 1
    shutil.move(src, dst)
    return dst


def _move_duplicates(similar, move_to):
    """Перемещает дубликаты (всё, кроме эталонного файла группы) в move_to."""
    os.makedirs(move_to, exist_ok=True)
    moved = 0
    freed = 0
    for paths in similar:
        if len(paths) < 2:
            continue
        keep = _pick_canonical(paths)
        for p in paths:
            if p == keep or not os.path.exists(p):
                continue
            try:
                size = os.path.getsize(p)
                _safe_move(p, move_to)
                moved += 1
                freed += size
            except Exception as e:
                print("  Не удалось переместить %s: %s" % (p, e))
    return moved, freed


def print_stats(similar_groups, files, failed_paths=None, failed_reasons=None):
    total_files = len(files)
    total_size = sum(f["size"] for f in files)

    similar_files = set()
    for group in similar_groups:
        similar_files.update(group)

    compared_files = total_files - len(similar_files)

    print("\n" + "=" * 60)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 60)
    print("Всего файлов в индексе: %d" % total_files)
    print("Общий размер: %.2f MB" % (total_size / (1024 ** 2)))

    print("\nФайлов в похожих группах: %d (%d%%)" % (len(similar_files), len(similar_files) * 100 // total_files if total_files else 0))
    print("Файлов без дубликатов: %d (%d%%)" % (compared_files, compared_files * 100 // total_files if total_files else 0))

    if failed_paths:
        print("\n⚠ Повреждённых/недоступных: %d (%d%%)" % (len(failed_paths), len(failed_paths) * 100 // total_files if total_files else 0))

    print("\n" + "=" * 60)
    print("ПОХОЖИЕ ИЗОБРАЖЕНИЯ (pHash + LSH)")
    print("=" * 60)
    print("Групп похожих: %d" % len(similar_groups))
    sim_count = sum(len(g) - 1 for g in similar_groups)
    print("Похожих файлов (сверх первого в группе): %d" % sim_count)
    if similar_groups:
        print("\nТоп-10 групп похожих изображений:")
        for i, group in enumerate(sorted(similar_groups, key=len, reverse=True)[:10]):
            print("  Группа %d (%d файлов):" % (i + 1, len(group)))
            canonical = _pick_canonical(group)
            for p in group[:5]:
                marker = "  [оригинал]" if p == canonical else ""
                print("    %s%s" % (p, marker))
            if len(group) > 5:
                print("    ... и ещё %d файлов" % (len(group) - 5))

    print("\n" + "=" * 60)
    print("ПОВРЕЖДЁННЫЕ/НЕДОСТУПНЫЕ ФАЙЛЫ")
    print("=" * 60)
    if failed_paths:
        print("Всего: %d файлов" % len(failed_paths))
        print("\nПолные пути:")
        for i, p in enumerate(failed_paths[:20], 1):
            reason = failed_reasons.get(p, "повреждён/нечитаемый") if failed_reasons else "повреждён/нечитаемый"
            print("  %d. %s  [%s]" % (i, p, reason))
        if len(failed_paths) > 20:
            print("  ... и ещё %d файлов" % (len(failed_paths) - 20))

        print("\nПричины ошибок:")
        reason_counts = {}
        for path in failed_paths:
            if failed_reasons:
                reason = failed_reasons.get(path, "повреждён/нечитаемый")
            else:
                reason = "повреждён/нечитаемый"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print("  %s: %d файлов" % (reason, count))
    else:
        print("OK Нет повреждённых или недоступных файлов")


def run(move_to=None, progress_callback=None, incremental=True):
    files = load_index()
    if files is None:
        print("Индекс не найден. Сначала выполните: python main.py scan --path ...")
        return None

    print("Обработка %d файлов..." % len(files))
    if progress_callback is not None:
        progress_callback("dedup", 0, len(files), "Подготовка к анализу")

    # Текущий mtime для каждого файла (из индекса)
    current_mtime_map = {f["path"]: f.get("mtime", 0) for f in files}

    # Определяем, какие файлы нужно обработать
    if incremental:
        # Загружаем существующие хэши с mtime
        existing_with_mtime = database.load_phashes_with_mtime() or {}
        existing_paths = set(existing_with_mtime.keys())
        current_paths = {f["path"] for f in files}

        # Файлы, для которых нужно вычислить pHash:
        #   - новые (нет в existing_paths)
        #   - изменённые (mtime отличается от сохранённого)
        files_to_process = []
        for f in files:
            p = f["path"]
            if p not in existing_paths:
                files_to_process.append(f)
            else:
                saved_mtime = existing_with_mtime[p][1]
                if f.get("mtime", 0) != saved_mtime:
                    files_to_process.append(f)

        print(f"Инкрементальный режим:")
        print(f"  Всего файлов: {len(files)}")
        print(f"  Новых/изменённых: {len(files_to_process)}")
        print(f"  Уже есть в кэше: {len(files) - len(files_to_process)}")
    else:
        files_to_process = files
        print("Полный режим: пересчёт всех хэшей")

    # Если нет файлов для обработки, показываем прогресс 100% и возвращаем существующие группы
    if not files_to_process:
        if progress_callback is not None:
            progress_callback("dedup", len(files), len(files), "Все файлы уже обработаны")
        existing_groups = database.load_dedup_groups() or []
        existing_failed = database.load_failed_paths() or []
        print("Нет новых файлов для дедупликации.")
        return existing_groups, list(existing_failed), {}

    similar, failed_paths, failed_reasons, hash_map = find_similar_images(
        files_to_process, progress_callback=progress_callback
    )

    # Сохраняем вычисленные хэши инкрементально (с mtime)
    if hash_map:
        mtime_map_for_save = {p: current_mtime_map.get(p, 0) for p in hash_map}
        if incremental:
            database.save_phashes_incremental(hash_map, mtime_map_for_save)
        else:
            # Полный режим: сохраняем все хэши (включая ранее вычисленные)
            all_hashes = dict(database.load_phashes() or {})
            all_hashes.update(hash_map)
            all_mtime = {p: current_mtime_map.get(p, 0) for p in all_hashes}
            database.save_phashes(all_hashes, all_mtime)

    # Если инкрементальный режим — объединяем с существующими группами
    if incremental and existing_with_mtime:
        # Загружаем существующие группы
        existing_groups = database.load_dedup_groups() or []
        # Объединяем: новые группы + существующие (которые не пересекаются с новыми)
        new_paths = set()
        for group in similar:
            new_paths.update(group)

        # Оставляем существующие группы, которые не содержат новых файлов
        preserved_groups = []
        for group in existing_groups:
            if not any(p in new_paths for p in group):
                preserved_groups.append(group)

        similar = preserved_groups + similar

    # Выбираем эталонный файл для каждой группы дубликатов.
    # Только эталоны (is_canonical=1) пойдут в эмбеддинг; дубликаты исключаются.
    canonical_paths = set()
    for group in similar:
        if len(group) >= 2:
            canonical_paths.add(_pick_canonical(group))

    database.save_dedup_groups(similar, canonical_paths)
    database.save_failed_files(failed_paths, failed_reasons, incremental=incremental)

    print_stats(similar, files, failed_paths, failed_reasons)

    total_files = len(files)
    similar_files = sum(len(group) for group in similar)
    similar_dup_files = sum(len(group) - 1 for group in similar)

    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)
    print("Обработано файлов: %d" % total_files)
    print("Похожих изображений: %d файлов в %d группах" % (similar_files, len(similar)))
    print("  (из них дубликаты для удаления: %d)" % similar_dup_files)
    if failed_paths:
        print("Повреждённых/недоступных: %d файлов" % len(failed_paths))

    accounted = similar_files + (len(failed_paths) if failed_paths else 0)
    remaining = total_files - accounted
    print("\nСводка:")
    print("  В похожих: %d файлов" % similar_files)
    print("  Повреждённых: %d файлов" % (len(failed_paths) if failed_paths else 0))
    print("  Без дубликатов: %d файлов" % remaining)
    print("  Итого: %d = %d" % (similar_files + (len(failed_paths) if failed_paths else 0) + remaining, total_files))
    print("\nМожно удалить: %d файлов (%d%% от общего)" % (
        similar_dup_files,
        similar_dup_files * 100 // total_files if total_files else 0))

    if move_to:
        print("\nПеремещение дубликатов в: %s" % move_to)
        moved, freed = _move_duplicates(similar, move_to)
        print("Перемещено файлов: %d" % moved)
        print("Освобождено: %.2f MB" % (freed / (1024 ** 2)))

    print("Результат сохранён в БД: %s" % config.DB_FILE)
    return similar, failed_paths, failed_reasons