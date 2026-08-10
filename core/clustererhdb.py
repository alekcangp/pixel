from collections import Counter
import warnings

import numpy as np
import umap
import hdbscan

import config
from core import database
from core import scanner

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_embeddings(exclude_duplicates=True):
    """Загружает эмбеддинги с опциональным исключением дубликатов.
    
    Args:
        exclude_duplicates: если True, исключает дубликаты (is_canonical=0).
    """
    return database.load_embeddings(exclude_duplicates=exclude_duplicates)


def _l2_normalize(embeddings):
    """L2-нормализация эмбеддингов для cosine metric."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _adaptive_min_cluster_size(n_samples):
    """Адаптивный min_cluster_size для Level 1 HDBSCAN.

    Масштабируется с размером датасета, чтобы число кластеров
    оставалось в разумных пределах (10-60) для датасетов 100-11000+ точек.

    Args:
        n_samples: int, количество точек в датасете.

    Returns:
        int, min_cluster_size.
    """
    if n_samples < 100:
        return max(3, n_samples // 20)       # 50-100  → 3-5
    elif n_samples < 1000:
        return max(5, n_samples // 50)       # 100-1000 → 5-20
    elif n_samples < 5000:
        return max(10, n_samples // 200)     # 1000-5000 → 10-25
    else:
        return config.HDBSCAN_MIN_CLUSTER_SIZE  # 5000+ → 25 (протокол)


def _adaptive_sub_min_cluster_size(mega_size):
    """Адаптивный min_cluster_size для Sub-HDBSCAN (Level 2).

    Масштабируется от размера мега-кластера, чтобы даже маленькие
    мега-кластеры (40-200 точек) давали осмысленные микро-кластеры.

    Args:
        mega_size: int, размер мега-кластера.

    Returns:
        int, min_cluster_size для Sub-HDBSCAN.
    """
    return max(5, min(config.SUB_HDBSCAN_MIN_CLUSTER_SIZE, mega_size // 10))


def _reduce_dimensions_umap(embeddings, n_neighbors=None, n_components=None, min_dist=None):
    """Снижение размерности через UMAP.
    
    Args:
        embeddings: np.ndarray, shape (n_samples, n_features)
        n_neighbors: int, размер локальной окрестности (None = авто)
        n_components: int, целевая размерность (None = из config)
        min_dist: float, минимальное расстояние между точками (None = из config)
    
    Returns:
        np.ndarray, shape (n_samples, n_components)
    """
    n_samples = len(embeddings)
    
    # Параметры по умолчанию из config
    if n_neighbors is None:
        n_neighbors = getattr(config, "UMAP_N_NEIGHBORS", 15)
    if n_components is None:
        n_components = getattr(config, "UMAP_N_COMPONENTS", 3)
    if min_dist is None:
        min_dist = getattr(config, "UMAP_MIN_DIST", 0.0)
    
    # Адаптация n_neighbors для очень маленьких датасетов
    if n_samples < 50:
        n_neighbors = max(5, min(n_neighbors, n_samples // 5))
    else:
        n_neighbors = max(10, min(n_neighbors, n_samples - 1))
    
    print(f"UMAP: n_samples={n_samples}, n_neighbors={n_neighbors}, "
          f"n_components={n_components}, min_dist={min_dist}")
    
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return np.zeros((n_samples, n_components))
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=getattr(config, "UMAP_METRIC", "cosine"),
        random_state=config.CLUSTER_RANDOM_STATE,
        low_memory=True,  # Для больших датасетов
    )
    
    embeddings_reduced = reducer.fit_transform(embeddings)
    return embeddings_reduced


def _explode_mega_cluster(raw_embeddings, mega_mask):
    """Level 2: Взрыв мега-кластера через Sub-UMAP + Sub-HDBSCAN (leaf).

    Args:
        raw_embeddings: np.ndarray, shape (N, 768) — raw SigLIP 2 векторы ВСЕХ точек.
        mega_mask: np.ndarray, shape (N,) bool — True для точек мега-кластера.

    Returns:
        sub_labels: np.ndarray, shape (N,) — метки для точек мега-кластера.
            (-1 = визуальный мусор, >0 = спасённые микро-кластеры).
            Для точек вне мега-кластера значение не определено (0).
    """
    mega_emb = raw_embeddings[mega_mask]
    mega_size = len(mega_emb)
    
    # L2-нормализация для cosine metric
    mega_emb = _l2_normalize(mega_emb)
    
    # Sub-UMAP: жёсткая локальная проекция для выявления микро-структур
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return np.full(mega_size, -1, dtype=int)
    sub_umap = umap.UMAP(
        n_neighbors=config.SUB_UMAP_N_NEIGHBORS,
        n_components=config.SUB_UMAP_N_COMPONENTS,
        min_dist=config.SUB_UMAP_MIN_DIST,
        metric=config.SUB_UMAP_METRIC,
        random_state=config.CLUSTER_RANDOM_STATE,
        low_memory=True,
    )
    print(f"Sub-UMAP: n_samples={mega_size}, n_neighbors={config.SUB_UMAP_N_NEIGHBORS}, "
          f"n_components={config.SUB_UMAP_N_COMPONENTS}, min_dist={config.SUB_UMAP_MIN_DIST}")
    sub_emb = sub_umap.fit_transform(mega_emb)
    
    # Sub-HDBSCAN: leaf — срезает пики, плоский фон → -1
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return np.full(mega_size, -1, dtype=int)
    sub_min_cluster_size = _adaptive_sub_min_cluster_size(mega_size)
    sub_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=sub_min_cluster_size,
        min_samples=config.SUB_HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=config.SUB_HDBSCAN_SELECTION_METHOD,
    )
    print(f"Sub-HDBSCAN: min_cluster_size={sub_min_cluster_size}, "
          f"min_samples={config.SUB_HDBSCAN_MIN_SAMPLES}, "
          f"selection_method={config.SUB_HDBSCAN_SELECTION_METHOD}")
    sub_labels = sub_clusterer.fit_predict(sub_emb)
    
    # Статистика
    n_micro = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    n_noise = np.sum(sub_labels == -1)
    print(f"Sub-HDBSCAN: {n_micro} микро-кластеров, {n_noise} точек мусора "
          f"({n_noise / mega_size * 100:.1f}%)")
    
    return sub_labels


def _noise_rescue(level1_umap, level1_labels, noise_mask):
    """Noise Rescue: присоединение шумовых точек к ближайшим кластерам.

    Когда HDBSCAN помечает слишком много точек как шум (-1), это часто
    означает, что параметры слишком строгие. Данная функция обучает
    KNeighborsClassifier на валидных кластерах и предсказывает метки
    для шумовых точек, но только если они достаточно близки к кластеру
    (расстояние <= NOISE_RESCUE_MAX_DIST).

    Args:
        level1_umap: np.ndarray, shape (N, 3) — UMAP-эмбеддинги всех точек.
        level1_labels: np.ndarray, shape (N,) — метки Level 1 (после refine).
        noise_mask: np.ndarray, shape (N,) bool — True для шумовых точек (-1).

    Returns:
        np.ndarray, shape (N,) — обновлённые метки (шум присоединён к кластерам).
    """
    from sklearn.neighbors import KNeighborsClassifier

    labels = level1_labels.copy()
    n_noise = int(noise_mask.sum())
    if n_noise == 0:
        return labels

    # Обучаем только на валидных кластерах (исключаем шум -1 и мусор -2)
    train_mask = (level1_labels > 0) & (~noise_mask)
    if not train_mask.any():
        return labels

    knn = KNeighborsClassifier(
        n_neighbors=config.KNN_N_NEIGHBORS,
        metric=config.KNN_METRIC,
    )
    knn.fit(level1_umap[train_mask], level1_labels[train_mask])

    # Предсказываем метки для шумовых точек
    noise_umap = level1_umap[noise_mask]
    pred_labels = knn.predict(noise_umap)

    # Проверяем расстояние до ближайшего соседа: если слишком далеко — оставляем шум
    dist, _ = knn.kneighbors(noise_umap, n_neighbors=1)
    max_dist = getattr(config, "NOISE_RESCUE_MAX_DIST", 0.5)
    close_mask = dist[:, 0] <= max_dist

    # Присоединяем только близкие шумовые точки
    noise_indices = np.where(noise_mask)[0]
    labels[noise_indices[close_mask]] = pred_labels[close_mask]

    n_rescued = int(close_mask.sum())
    print(f"Noise Rescue: {n_rescued} из {n_noise} шумовых точек присоединены к кластерам "
          f"({n_rescued / n_noise * 100:.1f}%)")

    return labels


def _knn_rescue(level1_umap, level1_labels, saved_mask, saved_umap, exclude_label=None):
    """KNN Rescue: переклассификация спасённых точек в глобальные кластеры.

    Обучает KNeighborsClassifier на Level 1 UMAP-эмбеддингах валидных
    кластеров (исключая шум -1, спасённые точки и исходный мега-кластер)
    и предсказывает глобальные метки для спасённых точек.

    Args:
        level1_umap: np.ndarray, shape (N, 3) — UMAP-эмбеддинги Level 1 всех точек.
        level1_labels: np.ndarray, shape (N,) — метки Level 1 (валидные кластеры >0).
        saved_mask: np.ndarray, shape (N,) bool — True для спасённых точек.
        saved_umap: np.ndarray, shape (M, 3) — UMAP-эмбеддинги спасённых точек.
        exclude_label: int или None — метка исходного мега-кластера,
            которую нужно исключить из обучения (чтобы спасённые не
            возвращались обратно в него).

    Returns:
        np.ndarray, shape (M,) — новые глобальные метки для спасённых точек.
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    # Обучаем только на валидных кластерах Level 1
    # (исключаем шум -1, спасённые точки и исходный мега-кластер)
    train_mask = (level1_labels > 0) & (~saved_mask)
    if exclude_label is not None:
        train_mask &= (level1_labels != exclude_label)
    
    knn = KNeighborsClassifier(
        n_neighbors=config.KNN_N_NEIGHBORS,
        metric=config.KNN_METRIC,
    )
    knn.fit(level1_umap[train_mask], level1_labels[train_mask])
    return knn.predict(saved_umap)


def refine_mega_cluster(embeddings, level1_labels, level1_umap):
    """Полный пайплайн Level 2 + KNN Rescue.

    Автоматически определяет ВСЕ мега-кластеры (размер > порога),
    взрывает каждый через Sub-UMAP + Sub-HDBSCAN (leaf), разделяет на
    визуальный мусор (-1) и спасённые микро-кластеры (>0), затем
    переклассифицирует спасённые точки в глобальные кластеры через KNN.

    Args:
        embeddings: np.ndarray, shape (N, 768) — raw SigLIP 2 векторы ВСЕХ точек.
        level1_labels: np.ndarray, shape (N,) — метки Level 1.
        level1_umap: np.ndarray, shape (N, 3) — UMAP-эмбеддинги Level 1.

    Returns:
        final_labels: np.ndarray, shape (N,) — итоговые метки.
            (-1 = шум Level 1, -2 = визуальный мусор, >0 = валидные кластеры).
        stats: dict — статистика {mega_ids, total_garbage, total_saved, total_micro}.
    """
    n = len(embeddings)
    final_labels = level1_labels.copy()
    
    # 1. Автоопределение ВСЕХ мега-кластеров: размер > порога
    counter = Counter(level1_labels)
    valid = {k: v for k, v in counter.items() if k > 0}
    if not valid:
        return final_labels, {"mega_ids": []}
    
    mega_ids = [k for k, v in valid.items() if v / n > config.MEGA_CLUSTER_THRESHOLD]
    mega_ids.sort(key=lambda k: -valid[k])  # по убыванию размера
    
    if not mega_ids:
        max_id = max(valid, key=valid.get)
        print(f"Мега-кластер не обнаружен (макс. кластер {max_id}: "
              f"{valid[max_id]} точек, {valid[max_id] / n * 100:.1f}% <= {config.MEGA_CLUSTER_THRESHOLD * 100:.0f}%)")
        return final_labels, {"mega_ids": []}
    
    total_garbage = 0
    total_saved = 0
    total_micro = 0
    
    for mega_id in mega_ids:
        if scanner.STOP_REQUESTED:
            print("\nОстановка кластеризации по запросу пользователя.")
            break
        mega_size = valid[mega_id]
        print(f"\n=== Level 2: Взрыв мега-кластера ===")
        print(f"Мега-кластер: ID={mega_id}, {mega_size} точек ({mega_size / n * 100:.1f}%)")
        
        mega_mask = level1_labels == mega_id
        
        # 2. Level 2: взрыв
        sub_labels = _explode_mega_cluster(embeddings, mega_mask)
        
        # 3. Разделение: -1 = мусор, >0 = спасённые
        # sub_labels имеет длину mega_size (только точки мега-кластера),
        # поэтому сопоставляем с полным массивом через mega_indices.
        mega_indices = np.where(mega_mask)[0]
        sub_labels_full = np.zeros(n, dtype=int)
        sub_labels_full[mega_indices] = sub_labels
        
        garbage_mask = sub_labels_full == -1
        # HDBSCAN может вернуть 0 как валидную метку микро-кластера,
        # поэтому спасаем все точки мега-кластера с меткой >= 0.
        saved_mask = mega_mask & (sub_labels_full >= 0)
        
        # 4. KNN Rescue: спасённые → глобальные кластеры
        # Исключаем исходный мега-кластер из обучения, чтобы спасённые
        # не возвращались обратно в него.
        if saved_mask.any():
            saved_umap = level1_umap[saved_mask]
            new_labels = _knn_rescue(level1_umap, level1_labels, saved_mask, saved_umap, exclude_label=mega_id)
            final_labels[saved_mask] = new_labels
        
        # 5. Мусор → GARBAGE_LABEL (-2)
        final_labels[garbage_mask] = config.GARBAGE_LABEL
        
        n_micro = len(set(sub_labels[sub_labels > 0]))
        n_garbage = int(garbage_mask.sum())
        n_saved = int(saved_mask.sum())
        total_garbage += n_garbage
        total_saved += n_saved
        total_micro += n_micro
        
        print(f"Level 2: {n_micro} микро-кластеров, "
              f"{n_garbage} мусора, {n_saved} спасённых")
        print(f"KNN Rescue: {n_saved} спасённых поглощены в глобальные кластеры")
    
    stats = {
        "mega_ids": mega_ids,
        "total_garbage": total_garbage,
        "total_saved": total_saved,
        "total_micro": total_micro,
    }
    
    return final_labels, stats


def cluster(embeddings, progress_callback=None):
    """Кластеризация через HDBSCAN с предварительным UMAP + Level 2.

    Args:
        embeddings: np.ndarray, shape (n_samples, n_features)
        progress_callback: callable(stage, current, total, message)

    Returns:
        np.ndarray, shape (n_samples,) — метки кластеров
            (-1 = шум, -2 = визуальный мусор, >0 = валидные кластеры).
    """
    n_samples = len(embeddings)
    
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)
    
    # Для очень маленьких датасетов fallback на KMeans
    if n_samples < 20:
        print(f"Мало данных ({n_samples}), используем KMeans fallback")
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(n_samples, config.CLUSTER_MIN_CLUSTERS)
        n_clusters = max(1, n_clusters)
        emb = _l2_normalize(embeddings)
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=config.CLUSTER_RANDOM_STATE,
            batch_size=256,
        )
        labels = km.fit_predict(emb)
        if progress_callback:
            progress_callback("cluster", 1, n_samples, "Мало данных, используем KMeans")
        return labels
    
    # Шаг 1: L2 нормализация
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return np.zeros(n_samples, dtype=int)
    if progress_callback:
        progress_callback("cluster", 0, n_samples, "Нормализация эмбеддингов...")
    embeddings = _l2_normalize(embeddings)
    
    # Шаг 2: Уменьшение размерности через UMAP
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return np.zeros(n_samples, dtype=int)
    if progress_callback:
        progress_callback("cluster", 0, n_samples, "UMAP: снижение размерности...")
    embeddings_reduced = _reduce_dimensions_umap(embeddings)
    
    # Шаг 3: Адаптивные параметры HDBSCAN
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return np.zeros(n_samples, dtype=int)
    min_cluster_size = _adaptive_min_cluster_size(n_samples)
    min_samples = max(2, min_cluster_size // 5)
    
    print(f"HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    # Шаг 4: Кластеризация
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return np.zeros(n_samples, dtype=int)
    if progress_callback:
        progress_callback("cluster", 0, n_samples, "HDBSCAN: инициализация...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # UMAP уже даёт cosine-like space
        cluster_selection_method=getattr(config, "HDBSCAN_SELECTION_METHOD", "eom"),
        prediction_data=True,  # Для возможности predict на новых данных
    )
    
    if progress_callback:
        progress_callback("cluster", 0, n_samples, "HDBSCAN: кластеризация...")
    
    labels = clusterer.fit_predict(embeddings_reduced)
    
    # Статистика Level 1
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"HDBSCAN нашёл {n_clusters_found} кластеров, {n_noise} точек шума "
          f"({n_noise / n_samples * 100:.1f}%)")
    
    # Шаг 5: Level 2 — взрыв мега-кластера + KNN Rescue
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return final_labels
    if progress_callback:
        progress_callback("cluster", 0, n_samples, "Level 2: взрыв мега-кластера...")
    
    final_labels, stats = refine_mega_cluster(embeddings, labels, embeddings_reduced)
    
    # Шаг 6: Noise Rescue — присоединение избыточного шума к ближайшим кластерам
    # Если HDBSCAN пометил слишком много точек как шум (> NOISE_RESCUE_THRESHOLD),
    # это значит параметры слишком строгие. Присоединяем шум к ближайшим кластерам.
    if scanner.STOP_REQUESTED:
        print("\nОстановка кластеризации по запросу пользователя.")
        return final_labels
    n_noise_after = int(np.sum(final_labels == -1))
    noise_threshold = getattr(config, "NOISE_RESCUE_THRESHOLD", 0.15)
    if n_noise_after / n_samples > noise_threshold:
        if progress_callback:
            progress_callback("cluster", 0, n_samples, "Noise Rescue: присоединение шума к кластерам...")
        
        noise_mask = final_labels == -1
        final_labels = _noise_rescue(embeddings_reduced, final_labels, noise_mask)
    
    if progress_callback:
        progress_callback("cluster", n_samples, n_samples, "Кластеризация завершена")
    
    return final_labels


def print_stats(labels, paths):
    """Вывод статистики по кластерам."""
    n = len(labels)
    counter = Counter(labels)
    n_clusters = len(counter) - (1 if -1 in counter else 0) - (1 if config.GARBAGE_LABEL in counter else 0)
    n_noise = counter.get(-1, 0)
    n_garbage = counter.get(config.GARBAGE_LABEL, 0)
    
    print(f"\nВсего точек: {n}")
    print(f"Кластеров: {n_clusters}")
    print(f"Точек шума: {n_noise} ({n_noise / n * 100:.1f}%)")
    print(f"Визуального мусора: {n_garbage} ({n_garbage / n * 100:.1f}%)")
    if n_clusters > 0:
        print(f"Средний размер кластера: {(n - n_noise - n_garbage) / n_clusters:.1f}")
    
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(paths[idx])
    
    print("\nТоп-10 кластеров по размеру:")
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
    # Исключаем шум и мусор из топа
    sorted_clusters = [(k, v) for k, v in sorted_clusters if k not in (-1, config.GARBAGE_LABEL)]
    
    for label, members in sorted_clusters[:10]:
        print(f"  Кластер {label} ({len(members)} файлов):")
        for p in members[:10]:
            print(f"    {p}")
    
    return clusters


def run(progress_callback=None):
    """Запуск полного пайплайна кластеризации."""
    if progress_callback is not None:
        progress_callback("cluster", 0, 1, "Загрузка эмбеддингов из БД...")
    
    embeddings, paths = load_embeddings()
    if embeddings is None:
        print("Эмбеддинги не найдены. Сначала выполните: python main.py embed")
        return None
    
    total = len(embeddings)
    if progress_callback is not None:
        progress_callback("cluster", 0, total, "Подготовка к кластеризации HDBSCAN")
    
    print(f"Кластеризация {total} точек через HDBSCAN + UMAP...")
    
    if progress_callback is not None:
        progress_callback("cluster", 1, total, "UMAP: снижение размерности...")
    
    labels = cluster(embeddings, progress_callback=progress_callback)
    
    if scanner.STOP_REQUESTED:
        print("\nКластеризация остановлена. Результаты не сохранены.")
        return None
    
    if progress_callback is not None:
        progress_callback("cluster", int(total * 0.8), total, "HDBSCAN завершён")
    
    clusters = print_stats(labels, paths)
    
    # Сохраняем кластеры без имён (автоматическое именование отключено)
    database.save_clusters(clusters, {})
    
    # Сохраняем визуальный мусор отдельно (cluster_id = GARBAGE_LABEL)
    garbage_paths = clusters.get(config.GARBAGE_LABEL, [])
    if garbage_paths:
        database.save_garbage(garbage_paths)
        print(f"Сохранено {len(garbage_paths)} файлов визуального мусора (cluster_id={config.GARBAGE_LABEL})")
    
    print(f"\nКластеры сохранены в БД: {config.DB_FILE}")
    
    return clusters