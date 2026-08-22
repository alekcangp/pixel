"""Дедупликация изображений по перцептивному хэшу (pHash, imagehash).

Вся логика дедупликации собрана в одном файле:
  - вычисление перцептивного хэша dHash (64-битный int);
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

import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

# 16-битная таблица popcount (фолбэк для numpy 1.x, где нет np.bitwise_count).
_POPCOUNT16 = np.array([bin(v).count("1") for v in range(1 << 16)], dtype=np.uint8)


def _hamming_distances(xor):
    """popcount для массива uint64 (XOR). numpy2 -> np.bitwise_count, иначе 16-битная таблица."""
    bc = getattr(np, "bitwise_count", None)
    if bc is not None:
        return bc(xor)
    return _POPCOUNT16[xor.view(np.uint16).reshape(-1, 4)].sum(axis=1)


def _default_phash_workers():
    """Auto threads for pHash (4..8 by cpu)."""
    cpu = os.cpu_count() or 4
    return max(4, min(cpu, 8))


import config
from core import database
from core import scanner


# ============================================================
# Перцептивный хэш (pHash, 64 бита)
# ============================================================

# Используется один фиксированный хэш — imagehash.phash (классический DCT-pHash).
# Он устойчив к масштабированию/перекодировке и не требует OpenCV (imagehash
# работает на PIL+numpy). Раньше здесь был фолбек на самодельный FFT-хэш,
# из-за которого «почти одинаковые» фото расходились по разным группам
# дедупликации. dHash пробовали, но он слишком слабо различает плоские
# изображения (маски CapCut и т.п.) и даёт гигантские цепочки-группы.

def _bits_to_int(bits):
    """Преобразует массив битов 0/1 в 64-битный int."""
    arr = np.asarray(bits).ravel()
    if arr.size == 0:
        return 0
    h = 0
    for bit in arr:
        h = (h << 1) | int(bit > 0)
    return int(h)


def compute_phash(path):
    """pHash (DCT, imagehash.phash). Возвращает 64-битный int или None.

    Возвращает None при повреждении/недоступности файла.
    Имя и сигнатура сохранены для совместимости с core/search.py.
    """
    try:
        import imagehash
        img = Image.open(path)
        try:
            database.update_image_size(path, img.width, img.height)
        except Exception:
            pass
        return _bits_to_int(imagehash.phash(img).hash)
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
        dists = _hamming_distances(np.bitwise_xor(h_arr, cand_hashes))

        results = []
        for j, d in zip(cand_list, dists):
            if d <= radius:
                results.append((int(d), j))
        results.sort(key=lambda x: x[0])
        return results

    def find_clusters(self, progress_callback=None):
        """Кластеризует все хэши через Union-Find.

        Возвращает список групп: [[id, ...], ...] (только группы размером > 1).
        """
        uf = UnionFind()
        n = self._n
        if n == 0:
            return []

        # Оптимизация: banding-подход.
        # Для каждого подблока строим инвертированный лист элементов с одинаковым значением.
        block_groups = defaultdict(list)
        for i, (_, h) in enumerate(self._items):
            offset = 0
            for b in range(self._B):
                size = self._block_sizes[b]
                shift = 64 - offset - size
                block_val = (h >> shift) & ((1 << size) - 1)
                block_groups[(b, block_val)].append(i)
                offset += size

        # pHash-дубликаты отличаются не более чем на threshold бит из 64. При B
        # подблоках кандидат обязан совпадать минимум с (B - threshold) подблоками.
        # Требуем >= 2 общих подблока: исключает случайные коллизии блоков,
        # не теряя настоящих дубликатов.
        min_bands = max(2, self._B - self._threshold)
        hashes_arr = np.array(self._hashes, dtype=np.uint64) if self._hashes else None
        processed = 0
        for i, (_, h) in enumerate(self._items):
            if scanner.STOP_REQUESTED:
                print("\nОстановка кластеризации pHash по запросу пользователя.")
                break

            # Считаем, со сколькими подблоками совпадает каждый элемент.
            common = defaultdict(int)
            offset = 0
            for b in range(self._B):
                size = self._block_sizes[b]
                shift = 64 - offset - size
                block_val = (h >> shift) & ((1 << size) - 1)
                for j in block_groups.get((b, block_val), ()):
                    common[j] += 1
                offset += size

            cand_list = [j for j, c in common.items() if c >= min_bands and j != i]
            if not cand_list:
                processed += 1
                if progress_callback is not None:
                    progress_callback("dedup", processed, n, "Кластеризация pHash")
                continue

            # Векторизованное вычисление Hamming distance.
            cand_hashes = hashes_arr[cand_list]
            h_arr = np.full(len(cand_list), h, dtype=np.uint64)
            dists = _hamming_distances(np.bitwise_xor(h_arr, cand_hashes))
            for j, d in zip(cand_list, dists):
                if d <= self._threshold:
                    uf.union(i, j)

            processed += 1
            if progress_callback is not None:
                progress_callback("dedup", processed, n, "Кластеризация pHash")

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


def find_similar_images(files, max_workers=None, progress_callback=None, existing_hashes=None):
    paths_to_process = [f["path"] for f in files]
    total = len(paths_to_process)

    if max_workers is None:
        max_workers = _default_phash_workers()

    if total == 0 and not existing_hashes:
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
            if scanner.STOP_REQUESTED:
                print("\nОстановка дедупликации по запросу пользователя.")
                # Отменяем ещё не запущенные задачи: иначе ThreadPoolExecutor
                # при выходе из контекста (shutdown(wait=True)) будет ждать их
                # завершения, и остановка не сработает.
                for f in futures:
                    f.cancel()
                break
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

    # Объединяем новые хэши с существующими для поиска дубликатов между новыми и старыми файлами
    combined_hashes = list(hashes)
    if existing_hashes:
        combined_hashes.extend(existing_hashes)

    n = len(combined_hashes)
    if n == 0:
        return [], failed_paths, failed_reasons, hash_map

    print("\nКластеризация %d хэшей..." % n)
    index = LSHIndex(combined_hashes, threshold)
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
    files = scanner.load_index()
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

        # Исключаем дубликаты и повреждённые файлы (аналогично embedder.py)
        dup_paths = database.load_duplicate_paths() or set()
        failed_paths = database.load_failed_paths() or set()
        excluded = dup_paths | failed_paths

        # Файлы, для которых нужно вычислить pHash:
        #   - новые (нет в existing_paths)
        #   - изменённые (mtime отличается от сохранённого)
        #   - не входят в excluded (дубликаты/повреждённые)
        files_to_process = []
        for f in files:
            p = f["path"]
            if p in excluded:
                continue
            if p not in existing_paths:
                files_to_process.append(f)
            else:
                saved_mtime = existing_with_mtime[p][1]
                if f.get("mtime", 0) != saved_mtime:
                    files_to_process.append(f)

        print(f"Инкрементальный режим:")
        print(f"  Всего файлов: {len(files)}")
        print(f"  Исключено (дубликаты/повреждённые): {len(excluded)}")
        print(f"  Новых/изменённых: {len(files_to_process)}")
        print(f"  Уже есть в кэше: {len(files) - len(files_to_process) - len(excluded)}")
    else:
        # В полном режиме обрабатываем ВСЕ файлы, включая дубликаты и повреждённые
        files_to_process = files[:]
        print("Полный режим: пересчёт всех хэшей")
        print(f"  Всего файлов для обработки: {len(files_to_process)}")

    # Если нет файлов для обработки, показываем прогресс 100% и возвращаем существующие группы
    if not files_to_process:
        if progress_callback is not None:
            progress_callback("dedup", len(files), len(files), "Все файлы уже обработаны")
        existing_groups = database.load_dedup_groups() or []
        existing_failed = database.load_failed_paths() or []
        print("Нет новых файлов для дедупликации.")
        return existing_groups, list(existing_failed), {}

    # В инкрементальном режиме подготавливаем существующие хэши для поиска дубликатов
    # между новыми/изменёнными файлами и уже имеющимися в БД.
    existing_hashes_for_lsh = []
    if incremental and existing_with_mtime:
        for p, (h, mtime) in existing_with_mtime.items():
            if p in current_paths and p not in excluded:
                existing_hashes_for_lsh.append((p, h))

    similar, failed_paths, failed_reasons, hash_map = find_similar_images(
        files_to_process,
        progress_callback=progress_callback,
        existing_hashes=existing_hashes_for_lsh,
    )

    if scanner.STOP_REQUESTED:
        print("\nДедупликация остановлена. Сохранение частичных результатов...")
        if hash_map:
            mtime_map_for_save = {p: current_mtime_map.get(p, 0) for p in hash_map}
            if incremental:
                database.save_phashes_incremental(hash_map, mtime_map_for_save)
            else:
                all_hashes = dict(database.load_phashes() or {})
                all_hashes.update(hash_map)
                all_mtime = {p: current_mtime_map.get(p, 0) for p in all_hashes}
                database.save_phashes(all_hashes, all_mtime)
        if failed_paths:
            database.save_failed_files(failed_paths, failed_reasons, incremental=incremental)
        print("Частичные результаты сохранены.")
        return None

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

    # Выбираем эталонный файл для каждой группы дубликатов.
    # Только эталоны (is_canonical=1) пойдут в эмбеддинг; дубликаты исключаются.
    canonical_paths = set()
    for group in similar:
        if len(group) >= 2:
            canonical_paths.add(_pick_canonical(group))

    # Сохранение групп. В инкрементальном режиме save_dedup_groups сам объединяет
    # новые группы с уже хранящимися в БД (группы, не пересекающиеся с новым list,
    # сохраняются), поэтому слияние не зависит от состояния кэша pHash и не грязное
    # между инкрементными прогонами.
    database.save_dedup_groups(similar, canonical_paths, incremental=incremental)
    database.save_failed_files(failed_paths, failed_reasons, incremental=incremental)

    # Пересчитываем статистику из БД — полный набор групп после merge-сохранения
    # (не только новые, чтобы отчёт отражал реальное состояние индекса).
    saved_groups = database.load_dedup_groups() or []
    print_stats(saved_groups, files, failed_paths, failed_reasons)

    total_files = len(files)
    similar_files = sum(len(group) for group in saved_groups)
    similar_dup_files = sum(len(group) - 1 for group in saved_groups)

    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)
    print("Обработано файлов: %d" % total_files)
    print("Похожих изображений: %d файлов в %d группах" % (similar_files, len(saved_groups)))
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
        moved, freed = _move_duplicates(saved_groups, move_to)
        print("Перемещено файлов: %d" % moved)
        print("Освобождено: %.2f MB" % (freed / (1024 ** 2)))

    print("Результат сохранён в БД: %s" % config.DB_FILE)
    return saved_groups, failed_paths, failed_reasons