"""SQLite-хранилище вместо JSON/npy-файлов.

Схема:
  images        — отсканированные файлы (path, size, ext, mtime, phash)
  embeddings    — эмбеддинги (image_id, vector BLOB)
  clusters      — кластеризация (image_id, cluster_id)
  dedup_groups  — группы дубликатов (group_id, image_id, is_canonical)
  failed_files  — повреждённые/недоступные файлы (path, reason)
"""

import os
import sqlite3

import config

# Флаг для отслеживания инициализации
_initialized = False

# SQLite INTEGER — знаковое 64-битное. pHash — unsigned 64-бит.
# Преобразуем: unsigned → signed при сохранении, signed → unsigned при загрузке.
_SIGNED64 = 1 << 63


def _to_signed(val):
    if val >= _SIGNED64:
        return val - (1 << 64)
    return val


def _to_unsigned(val):
    if val < 0:
        return val + (1 << 64)
    return val


def _get_conn():
    """Открывает соединение и гарантирует, что схема создана."""
    global _initialized
    os.makedirs(os.path.dirname(config.DB_FILE), exist_ok=True)
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if not _initialized:
        _init_schema(conn)
        _initialized = True
    return conn


def _path_to_id_map(conn, paths):
    """Возвращает {path: id} одним батч-запросом (устраняет N+1-опрашивание)."""
    if not paths:
        return {}
    ids = {}
    step = 900  # лимит SQLite на число переменных ~999
    for i in range(0, len(paths), step):
        chunk = paths[i:i + step]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT path, id FROM images WHERE path IN ({placeholders})", chunk
        ).fetchall()
        ids.update({r["path"]: r["id"] for r in rows})
    return ids


def _init_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            path  TEXT UNIQUE NOT NULL,
            size  INTEGER NOT NULL DEFAULT 0,
            ext   TEXT NOT NULL DEFAULT '',
            mtime REAL NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT '',
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS image_hashes (
            image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            phash    INTEGER NOT NULL,
            mtime    REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            vector   BLOB NOT NULL,
            mtime    REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS clusters (
            image_id  INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            cluster_id INTEGER NOT NULL,
            name TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS dedup_groups (
            group_id    INTEGER NOT NULL,
            image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            is_canonical INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (group_id, image_id)
        );

        CREATE TABLE IF NOT EXISTS failed_files (
            path   TEXT PRIMARY KEY,
            reason TEXT NOT NULL DEFAULT 'corrupt'
        );

        CREATE TABLE IF NOT EXISTS selected_files (
            path   TEXT PRIMARY KEY,
            scope  TEXT NOT NULL DEFAULT 'global'
        );

        CREATE TABLE IF NOT EXISTS thumbnails (
            path  TEXT PRIMARY KEY,
            mtime REAL NOT NULL DEFAULT 0,
            size  INTEGER NOT NULL DEFAULT 0,
            ext   TEXT NOT NULL DEFAULT 'webp',
            blob  BLOB NOT NULL
        );
        """
    )
    # Миграция: добавляем колонки, если таблица создана без них (старая БД)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()]
    if "mtime" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN mtime REAL NOT NULL DEFAULT 0")
    if "source" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN source TEXT NOT NULL DEFAULT ''")

    cols = [r[1] for r in conn.execute("PRAGMA table_info(dedup_groups)").fetchall()]
    if "is_canonical" not in cols:
        conn.execute("ALTER TABLE dedup_groups ADD COLUMN is_canonical INTEGER NOT NULL DEFAULT 0")

    # Миграция: добавляем mtime в image_hashes и embeddings (старая БД)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(image_hashes)").fetchall()]
    if "mtime" not in cols:
        conn.execute("ALTER TABLE image_hashes ADD COLUMN mtime REAL NOT NULL DEFAULT 0")

    cols = [r[1] for r in conn.execute("PRAGMA table_info(embeddings)").fetchall()]
    if "mtime" not in cols:
        conn.execute("ALTER TABLE embeddings ADD COLUMN mtime REAL NOT NULL DEFAULT 0")
    conn.commit()


# ============================================================
# Selected files (выделенные файлы)
# ============================================================

def save_selected_files(paths, scope="global"):
    """Сохраняет выделенные файлы. paths: list[str], scope: str."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM selected_files WHERE scope = ?", (scope,))
        if paths:
            rows = [(p, scope) for p in paths]
            conn.executemany(
                "INSERT OR REPLACE INTO selected_files (path, scope) VALUES (?, ?)", rows
            )
        conn.commit()
    finally:
        conn.close()


def load_selected_files(scope="global"):
    """Возвращает set путей выделенных файлов или None."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT path FROM selected_files WHERE scope = ?", (scope,)).fetchall()
        if not rows:
            return None
        return {r["path"] for r in rows}
    finally:
        conn.close()


def get_selected_files_stats(scope="global"):
    """Возвращает (количество, суммарный размер в байтах) выбранных файлов.

    Данные берутся из БД (join selected_files -> images), без обращения
    к файловой системе. Это мгновенно, даже если файлы лежат на внешнем диске.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(i.size), 0)
            FROM selected_files s
            JOIN images i ON i.path = s.path
            WHERE s.scope = ?
            """,
            (scope,),
        ).fetchone()
        return (row[0], row[1]) if row else (0, 0)
    finally:
        conn.close()


# ============================================================
# Thumbnails (миниатюры WebP, BLOB)
# ============================================================

def save_thumbnail(path, mtime, size, blob_bytes, ext="webp"):
    """Сохраняет BLOB миниатюры (квадратный WebP) в таблицу thumbnails.

    Ключ — путь; запись валидна только пока совпадают mtime/size исходного
    файла (иначе миниатюра устарела и при чтении будет перегенерирована).
    """
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO thumbnails (path, mtime, size, ext, blob) VALUES (?, ?, ?, ?, ?)",
            (path, float(mtime), int(size), ext, sqlite3.Binary(blob_bytes)),
        )
        conn.commit()
    finally:
        conn.close()


def load_thumbnail(path, mtime, size):
    """Возвращает bytes BLOB миниатюры, если она актуальна (mtime/size совпадают), иначе None."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT blob FROM thumbnails WHERE path = ? AND mtime = ? AND size = ?",
            (path, float(mtime), int(size)),
        ).fetchone()
        return bytes(row[0]) if row else None
    finally:
        conn.close()


def load_thumbnails_for_paths(paths, mtime_sizes):
    """Батч-чтение миниатюр для списка путей (одним SQL-запросом).

    Args:
        paths: list[str] — пути.
        mtime_sizes: dict {path: (mtime, size)} — актуальные метаданные файлов;
            записи, у которых метаданные разошлись (файл изменён), пропускаются.

    Returns:
        dict {path: bytes}
    """
    if not paths:
        return {}
    conn = _get_conn()
    try:
        placeholders = ",".join("?" for _ in paths)
        rows = conn.execute(
            f"SELECT path, mtime, size, blob FROM thumbnails WHERE path IN ({placeholders})",
            list(paths),
        ).fetchall()
        result = {}
        for r in rows:
            expected = mtime_sizes.get(r["path"])
            if expected is None:
                continue
            exp_mtime, exp_size = expected
            if float(exp_mtime) == float(r["mtime"]) and int(exp_size) == int(r["size"]):
                result[r["path"]] = bytes(r["blob"])
        return result
    finally:
        conn.close()


def delete_thumbnail(path):
    """Удаляет строку миниатюры по пути."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM thumbnails WHERE path = ?", (path,))
        conn.commit()
    finally:
        conn.close()


def clear_thumbnails():
    """Полностью очищает таблицу миниатюр."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM thumbnails")
        conn.commit()
    finally:
        conn.close()


# ============================================================
# Images (сканирование)
# ============================================================

def save_images(files):
    """Полностью заменяет содержимое таблицы images и очищает производные таблицы."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM image_hashes")
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM clusters")
        conn.execute("DELETE FROM dedup_groups")
        conn.execute("DELETE FROM failed_files")
        conn.execute("DELETE FROM images")
        conn.executemany(
            "INSERT OR IGNORE INTO images (path, size, ext, mtime, source) VALUES (?, ?, ?, ?, ?)",
            [(f["path"], f.get("size", 0), f.get("ext", ""), f.get("mtime", 0), f.get("source", "")) for f in files],
        )
        conn.commit()
    finally:
        conn.close()


def load_images():
    """Возвращает список dict [{path, size, ext, mtime, source}] в порядке id, или None если таблица пуста."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT path, size, ext, mtime, source FROM images ORDER BY id").fetchall()
        if not rows:
            return None
        return [{"path": r["path"], "size": r["size"], "ext": r["ext"], "mtime": r["mtime"], "source": r["source"]} for r in rows]
    finally:
        conn.close()


def get_existing_images():
    """Возвращает dict {path: {id, size, ext, mtime, source}} для всех изображений в БД."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT id, path, size, ext, mtime, source FROM images").fetchall()
        return {r["path"]: {"id": r["id"], "size": r["size"], "ext": r["ext"], "mtime": r["mtime"], "source": r["source"]} for r in rows}
    finally:
        conn.close()


def update_image_size(path: str, width: int, height: int) -> None:
    """Сохраняет размер изображения в таблице images."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE images SET width = ?, height = ? WHERE path = ?",
            (int(width), int(height), path),
        )
        conn.commit()
    finally:
        conn.close()


def get_image_size(path: str):
    """Возвращает (width, height) из images или None."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT width, height FROM images WHERE path = ?", (path,)).fetchone()
        if row and row["width"] and row["height"]:
            return int(row["width"]), int(row["height"])
        return None
    finally:
        conn.close()


def count_missing_thumbnails() -> int:
    """Возвращает количество изображений без актуальной миниатюры."""
    conn = _get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM images i
            LEFT JOIN thumbnails t ON i.path = t.path
               AND t.mtime = i.mtime
               AND t.size = i.size
            WHERE i.path IS NOT NULL AND t.path IS NULL
            """
        ).fetchone()
        return int(row["cnt"]) if row else 0
    finally:
        conn.close()


def missing_thumbnail_paths(limit: int = None):
    """Возвращает список путей без актуальной миниатюры."""
    query = """
        SELECT i.path
        FROM images i
        LEFT JOIN thumbnails t ON i.path = t.path
           AND t.mtime = i.mtime
           AND t.size = i.size
        WHERE i.path IS NOT NULL AND t.path IS NULL
    """
    args = []
    if limit is not None:
        query += " LIMIT ?"
        args.append(int(limit))
    conn = _get_conn()
    try:
        return [r["path"] for r in conn.execute(query, args).fetchall()]
    finally:
        conn.close()


def update_images_incremental(new_files, changed_files, removed_paths, source=""):
    """Инкрементальное обновление таблицы images.

    new_files: list[dict] — новые файлы (path, size, ext, mtime)
    changed_files: list[dict] — изменённые файлы (path, size, ext, mtime)
    removed_paths: list[str] — пути удалённых файлов (только для текущего источника)
    source: str — источник (путь сканирования)
    """
    conn = _get_conn()
    try:
        # Удаляем исчезнувшие файлы ТОЛЬКО для текущего источника
        if removed_paths:
            conn.executemany(
                "DELETE FROM images WHERE path = ? AND source = ?",
                [(p, source) for p in removed_paths],
            )

        # Обновляем изменённые файлы
        if changed_files:
            conn.executemany(
                "UPDATE images SET size = ?, ext = ?, mtime = ?, source = ? WHERE path = ?",
                [(f["size"], f["ext"], f["mtime"], source, f["path"]) for f in changed_files],
            )

        # Добавляем новые файлы
        if new_files:
            conn.executemany(
                "INSERT OR IGNORE INTO images (path, size, ext, mtime, source) VALUES (?, ?, ?, ?, ?)",
                [(f["path"], f.get("size", 0), f.get("ext", ""), f.get("mtime", 0), source) for f in new_files],
            )

        conn.commit()
    finally:
        conn.close()


# ============================================================
# pHash
# ============================================================

def save_phashes(hash_map, mtime_map=None):
    """Сохраняет phash для изображений. hash_map: {path: phash_int}, mtime_map: {path: mtime}."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM image_hashes")
        rows = []
        id_map = _path_to_id_map(conn, list(hash_map.keys()))
        for path, phash in hash_map.items():
            image_id = id_map.get(path)
            if image_id is not None:
                mtime = mtime_map.get(path, 0) if mtime_map else 0
                rows.append((image_id, _to_signed(int(phash)), float(mtime)))
        conn.executemany(
            "INSERT OR REPLACE INTO image_hashes (image_id, phash, mtime) VALUES (?, ?, ?)", rows
        )
        conn.commit()
    finally:
        conn.close()


def save_phashes_incremental(hash_map, mtime_map=None):
    """Инкрементальное сохранение phash. hash_map: {path: phash_int}, mtime_map: {path: mtime}."""
    conn = _get_conn()
    try:
        rows = []
        id_map = _path_to_id_map(conn, list(hash_map.keys()))
        for path, phash in hash_map.items():
            image_id = id_map.get(path)
            if image_id is not None:
                mtime = mtime_map.get(path, 0) if mtime_map else 0
                rows.append((image_id, _to_signed(int(phash)), float(mtime)))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO image_hashes (image_id, phash, mtime) VALUES (?, ?, ?)", rows
            )
            conn.commit()
    finally:
        conn.close()


def load_phashes():
    """Возвращает dict {path: phash_int} или None если пусто."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT i.path, h.phash FROM image_hashes h JOIN images i ON h.image_id = i.id"
        ).fetchall()
        if not rows:
            return None
        return {r["path"]: _to_unsigned(int(r["phash"])) for r in rows}
    finally:
        conn.close()


def load_phashes_with_mtime():
    """Возвращает dict {path: (phash_int, mtime)} или None если пусто."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT i.path, h.phash, h.mtime FROM image_hashes h JOIN images i ON h.image_id = i.id"
        ).fetchall()
        if not rows:
            return None
        return {r["path"]: (_to_unsigned(int(r["phash"])), float(r["mtime"])) for r in rows}
    finally:
        conn.close()


# ============================================================
# Embeddings
# ============================================================

def save_embeddings(vectors, paths, mtime_map=None, incremental=False):
    """Сохраняет эмбеддинги. vectors: np.ndarray (N, D), paths: list[str], mtime_map: {path: mtime}.
    
    Args:
        incremental: если True, не удаляет старые эмбеддинги, а обновляет/добавляет новые.
    """
    import numpy as np

    conn = _get_conn()
    try:
        if not incremental:
            conn.execute("DELETE FROM embeddings")
        rows = []
        id_map = _path_to_id_map(conn, list(paths))
        for vec, path in zip(vectors, paths):
            image_id = id_map.get(path)
            if image_id is not None:
                blob = np.asarray(vec, dtype=np.float32).tobytes()
                mtime = mtime_map.get(path, 0) if mtime_map else 0
                rows.append((image_id, blob, float(mtime)))
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (image_id, vector, mtime) VALUES (?, ?, ?)", rows
        )
        conn.commit()
    finally:
        conn.close()


def load_embeddings(exclude_duplicates=True):
    """Возвращает (np.ndarray (N,D), list[paths]) или (None, None).
    
    Args:
        exclude_duplicates: если True, исключает дубликаты (is_canonical=0) из загрузки.
    """
    import numpy as np

    conn = _get_conn()
    try:
        if exclude_duplicates:
            # Исключаем дубликаты: берем только canonical изображения
            rows = conn.execute(
                """SELECT i.path, e.vector 
                   FROM embeddings e 
                   JOIN images i ON e.image_id = i.id 
                   LEFT JOIN dedup_groups d ON i.id = d.image_id AND d.is_canonical = 0
                   WHERE d.image_id IS NULL
                   ORDER BY i.id"""
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT i.path, e.vector FROM embeddings e JOIN images i ON e.image_id = i.id ORDER BY i.id"
            ).fetchall()
        if not rows:
            return None, None
        paths = [r["path"] for r in rows]
        vectors = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        return vectors, paths
    finally:
        conn.close()


def load_embeddings_with_mtime():
    """Возвращает (np.ndarray (N,D), list[paths], dict {path: mtime}) или (None, None, None)."""
    import numpy as np

    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT i.path, e.vector, e.mtime FROM embeddings e JOIN images i ON e.image_id = i.id ORDER BY i.id"
        ).fetchall()
        if not rows:
            return None, None, None
        paths = [r["path"] for r in rows]
        vectors = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        mtime_map = {r["path"]: float(r["mtime"]) for r in rows}
        return vectors, paths, mtime_map
    finally:
        conn.close()


# ============================================================
# Clusters
# ============================================================

def save_clusters(cluster_dict, cluster_names=None):
    """Сохраняет кластеры. cluster_dict: {cluster_id: [path, ...]}, cluster_names: {cluster_id: name}."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM clusters")
        rows = []
        all_paths = []
        for paths in cluster_dict.values():
            all_paths.extend(paths)
        id_map = _path_to_id_map(conn, all_paths)
        for cluster_id, paths in cluster_dict.items():
            # Получаем имя кластера если есть
            name = ""
            if cluster_names and cluster_id in cluster_names:
                name = cluster_names[cluster_id]

            for path in paths:
                image_id = id_map.get(path)
                if image_id is not None:
                    rows.append((image_id, int(cluster_id), name))
        conn.executemany(
            "INSERT OR REPLACE INTO clusters (image_id, cluster_id, name) VALUES (?, ?, ?)", rows
        )
        conn.commit()
    finally:
        conn.close()


def load_clusters():
    """Возвращает dict {cluster_id: [path, ...]} или None."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT i.path, c.cluster_id, c.name FROM clusters c JOIN images i ON c.image_id = i.id ORDER BY c.cluster_id"
        ).fetchall()
        if not rows:
            return None
        clusters = {}
        for r in rows:
            clusters.setdefault(int(r["cluster_id"]), []).append(r["path"])
        return clusters
    finally:
        conn.close()


def save_garbage(paths):
    """Помечает пути как визуальный мусор (cluster_id = GARBAGE_LABEL).

    Args:
        paths: list[str] — пути файлов, подтверждённых как визуальный мусор.
    """
    conn = _get_conn()
    try:
        rows = []
        for path in paths:
            r = conn.execute("SELECT id FROM images WHERE path = ?", (path,)).fetchone()
            if r:
                rows.append((r["id"], config.GARBAGE_LABEL, "garbage"))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO clusters (image_id, cluster_id, name) VALUES (?, ?, ?)", rows
            )
            conn.commit()
    finally:
        conn.close()


def load_clusters_with_names():
    """Возвращает (clusters dict, cluster_names dict) или (None, None)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT i.path, c.cluster_id, c.name FROM clusters c JOIN images i ON c.image_id = i.id ORDER BY c.cluster_id"
        ).fetchall()
        if not rows:
            return None, None
        clusters = {}
        cluster_names = {}
        for r in rows:
            cluster_id = int(r["cluster_id"])
            clusters.setdefault(cluster_id, []).append(r["path"])
            # Сохраняем имя кластера (все строки с одинаковым cluster_id имеют одинаковое name)
            if r["name"] and cluster_id not in cluster_names:
                cluster_names[cluster_id] = r["name"]
        return clusters, cluster_names
    finally:
        conn.close()


# ============================================================
# Dedup groups
# ============================================================

def save_dedup_groups(groups, canonical_paths=None, incremental=False):
    """Сохраняет группы дубликатов.

    groups:          list[list[str]] (каждый — список путей).
    canonical_paths: set/list путей-эталонов (is_canonical=1).

    При incremental=True старые группы, не пересекающиеся с новым списком
    (вычисляются по путям), сохраняются — глава таблицы не стирается вслепую.
    Это исключает риск потери групп, когда инкрементальный прогон обработал
    только новые/изменённые файлы, а кэш pHash пуст/частичен.
    """
    conn = _get_conn()
    try:
        if canonical_paths is None:
            canonical_set = None
        else:
            canonical_set = set(canonical_paths)

        # path → id одним запросом вместо N отдельных SELECT
        path_to_id = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM images").fetchall()}

        if incremental:
            # Загружаем существующие группы (по групповым ID), чтобы не потерять
            # те, что не покрыты новым списком (например, при частичной переиндексации).
            existing_rows = conn.execute(
                "SELECT i.path, g.group_id, g.is_canonical "
                "FROM dedup_groups g JOIN images i ON g.image_id = i.id "
                "ORDER BY g.group_id, i.id"
            ).fetchall()
            existing_groups = {}
            for r in existing_rows:
                existing_groups.setdefault(int(r["group_id"]), []).append(r["path"])
        else:
            existing_groups = {}

        # Пути, входящие в новые группы, — считаем покрытыми.
        new_paths = set()
        for group in groups:
            for p in group:
                new_paths.add(p)

        # Собираем пункт окончательного списка групп:
        # сначала новые (не пересекающиеся с существующими сохраняются), потом сохранённые старые.
        merged = [g for g in groups]
        for gid, paths in existing_groups.items():
            if not any(p in new_paths for p in paths):
                merged.append(paths)

        conn.execute("DELETE FROM dedup_groups")
        rows = []
        for group_id, paths in enumerate(merged):
            for path in paths:
                image_id = path_to_id.get(path)
                if image_id is None:
                    continue
                if canonical_set is None:
                    is_can = 1
                elif path in canonical_set:
                    is_can = 1
                elif len(paths) == 1:
                    is_can = 1
                else:
                    is_can = 0
                rows.append((group_id, image_id, is_can))
        conn.executemany(
            "INSERT OR REPLACE INTO dedup_groups (group_id, image_id, is_canonical) VALUES (?, ?, ?)", rows
        )
        conn.commit()
    finally:
        conn.close()


def load_dedup_groups():
    """Возвращает list[list[str]] или None."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT i.path, g.group_id FROM dedup_groups g JOIN images i ON g.image_id = i.id ORDER BY g.group_id, i.id"
        ).fetchall()
        if not rows:
            return None
        groups = {}
        for r in rows:
            groups.setdefault(int(r["group_id"]), []).append(r["path"])
        return list(groups.values())
    finally:
        conn.close()


def load_duplicate_paths():
    """Возвращает set путей-дубликатов (is_canonical=0) или None, если таблица пуста."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT i.path FROM dedup_groups g JOIN images i ON g.image_id = i.id WHERE g.is_canonical = 0"
        ).fetchall()
        if not rows:
            return None
        return {r["path"] for r in rows}
    finally:
        conn.close()


def get_duplicates_size():
    """Возвращает суммарный размер дубликатов (is_canonical=0) в байтах."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(i.size), 0) FROM dedup_groups g JOIN images i ON g.image_id = i.id WHERE g.is_canonical = 0"
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


# ============================================================
# Failed files (повреждённые/недоступные)
# ============================================================

def save_failed_files(failed_paths, failed_reasons=None, incremental=False):
    """Сохраняет повреждённые/недоступные файлы. failed_paths: list[str].

    При incremental=True объединяет с существующими (не стирает предыдущие).
    """
    conn = _get_conn()
    try:
        if not incremental:
            conn.execute("DELETE FROM failed_files")
        if failed_paths:
            rows = []
            for path in failed_paths:
                reason = "corrupt"
                if failed_reasons:
                    reason = failed_reasons.get(path, "corrupt")
                rows.append((path, reason))
            conn.executemany(
                "INSERT OR REPLACE INTO failed_files (path, reason) VALUES (?, ?)", rows
            )
            conn.commit()
    finally:
        conn.close()


def load_failed_paths():
    """Возвращает set путей повреждённых/недоступных файлов или None, если таблица пуста."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT path FROM failed_files").fetchall()
        if not rows:
            return None
        return {r["path"] for r in rows}
    finally:
        conn.close()


# ============================================================
# Очистка
# ============================================================

def clear_all():
    """Полностью очищает все таблицы БД."""
    conn = _get_conn()
    try:
        for table in ["images", "image_hashes", "embeddings", "clusters", "dedup_groups", "failed_files", "selected_files", "thumbnails"]:
            conn.execute("DELETE FROM %s" % table)
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


def delete_images_by_paths(paths):
    """Удаляет изображения и связанные миниатюры по списку путей."""
    if not paths:
        return
    conn = _get_conn()
    try:
        step = 900
        for i in range(0, len(paths), step):
            chunk = paths[i:i + step]
            placeholders = ",".join("?" for _ in chunk)
            conn.execute(f"DELETE FROM thumbnails WHERE path IN ({placeholders})", chunk)
            conn.execute(f"DELETE FROM images WHERE path IN ({placeholders})", chunk)
        conn.commit()
    finally:
        conn.close()


def load_db_stats():
    """Возвращает статистику: total, duplicates, unique, clusters, garbage, total_size."""
    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        duplicates = conn.execute(
            "SELECT COUNT(*) FROM dedup_groups g JOIN images i ON g.image_id = i.id WHERE g.is_canonical = 0"
        ).fetchone()[0]
        clusters = conn.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM clusters WHERE cluster_id > 0"
        ).fetchone()[0]
        garbage = conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE cluster_id = ?", (config.GARBAGE_LABEL,)
        ).fetchone()[0]
        total_size = conn.execute("SELECT COALESCE(SUM(size), 0) FROM images").fetchone()[0]
        return {
            "total": total,
            "duplicates": duplicates,
            "unique": total - duplicates,
            "clusters": clusters,
            "garbage": garbage,
            "total_size": total_size,
        }
    finally:
        conn.close()


def has_pending_dedup():
    """Возвращает True, если есть изображения без pHash и не отмеченные как повреждённые.

    Используется для скрытия неинформативной статистики уникальных/дубликатов,
    когда дедупликация не была завершена для всех файлов.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM images i
            LEFT JOIN image_hashes h ON i.id = h.image_id
            LEFT JOIN failed_files f ON i.path = f.path
            WHERE h.image_id IS NULL AND f.path IS NULL
            """
        ).fetchone()
        return row[0] > 0 if row else False
    finally:
        conn.close()
