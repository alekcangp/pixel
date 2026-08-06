from collections import Counter
import warnings

import numpy as np
import torch
import umap
import hdbscan
from sklearn.metrics import davies_bouldin_score, silhouette_score

import config
from core import database

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Предопределенные категории для автоименования
PREDEFINED_CATEGORIES = [
    "Природа", "Город", "Люди", "Животные", "Еда",
    "Архитектура", "Пейзаж", "Портрет", "Улица", "Интерьер",
    "Техника", "Спорт", "Искусство", "Транспорт", "Цветы",
    "Море", "Горы", "Лес", "Небо", "Дорога"
]

# Английские эквиваленты для эмбеддингов
CATEGORY_ENGLISH = {
    "Природа": "nature",
    "Город": "city",
    "Люди": "people",
    "Животные": "animals",
    "Еда": "food",
    "Архитектура": "architecture",
    "Пейзаж": "landscape",
    "Портрет": "portrait",
    "Улица": "street",
    "Интерьер": "interior",
    "Техника": "technology",
    "Спорт": "sport",
    "Искусство": "art",
    "Транспорт": "transport",
    "Цветы": "flowers",
    "Море": "sea",
    "Горы": "mountains",
    "Лес": "forest",
    "Небо": "sky",
    "Дорога": "road"
}


def load_embeddings():
    return database.load_embeddings()


def _l2_normalize(embeddings):
    """L2-нормализация эмбеддингов для cosine metric."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


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
    if n_components is None:
        n_components = getattr(config, "UMAP_N_COMPONENTS", 15)
    if min_dist is None:
        min_dist = getattr(config, "UMAP_MIN_DIST", 0.1)
    
    # Динамический n_neighbors в зависимости от объёма данных
    if n_neighbors is None:
        if n_samples < 50:
            n_neighbors = max(5, n_samples // 5)
        else:
            n_neighbors = max(10, int(np.sqrt(n_samples) / 2))
    
    # Ограничиваем n_neighbors количеством samples
    n_neighbors = min(n_neighbors, n_samples - 1)
    
    print(f"UMAP: n_samples={n_samples}, n_neighbors={n_neighbors}, "
          f"n_components={n_components}, min_dist={min_dist}")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=config.CLUSTER_RANDOM_STATE,
        low_memory=True,  # Для больших датасетов
    )
    
    embeddings_reduced = reducer.fit_transform(embeddings)
    return embeddings_reduced


def cluster(embeddings, n_clusters=None):
    """Кластеризация через HDBSCAN с предварительным UMAP.
    
    Args:
        embeddings: np.ndarray, shape (n_samples, n_features)
        n_clusters: int, игнорируется (оставлен для совместимости API)
    
    Returns:
        np.ndarray, shape (n_samples,) — метки кластеров (-1 = шум)
    """
    n_samples = len(embeddings)
    
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)
    
    # Для очень маленьких датасетов fallback на KMeans
    if n_samples < 20:
        print(f"Мало данных ({n_samples}), используем KMeans fallback")
        from core.clusterer import cluster as kmeans_cluster
        return kmeans_cluster(embeddings)
    
    # Шаг 1: L2 нормализация
    embeddings = _l2_normalize(embeddings)
    
    # Шаг 2: Уменьшение размерности через UMAP
    embeddings_reduced = _reduce_dimensions_umap(embeddings)
    
    # Шаг 3: Динамические параметры HDBSCAN
    min_cluster_size = getattr(config, "HDBSCAN_MIN_CLUSTER_SIZE", 5)
    min_samples = getattr(config, "HDBSCAN_MIN_SAMPLES", 3)
    
    # Адаптация под размер датасета
    if n_samples < 100:
        min_cluster_size = max(3, n_samples // 20)
    elif n_samples > 5000:
        min_cluster_size = max(10, n_samples // 100)
    else:
        min_cluster_size = max(5, n_samples // 50)
    
    min_samples = max(2, min_cluster_size // 3)
    
    print(f"HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    # Шаг 4: Кластеризация
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # UMAP уже даёт cosine-like space
        cluster_selection_method="eom",  # Excess of Mass - лучше для переменной плотности
        prediction_data=True,  # Для возможности predict на новых данных
    )
    
    labels = clusterer.fit_predict(embeddings_reduced)
    
    # Статистика
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"HDBSCAN нашёл {n_clusters_found} кластеров, {n_noise} точек шума "
          f"({n_noise / n_samples * 100:.1f}%)")
    
    return labels


def print_stats(labels, paths):
    """Вывод статистики по кластерам."""
    n = len(labels)
    counter = Counter(labels)
    n_clusters = len(counter) - (1 if -1 in counter else 0)
    n_noise = counter.get(-1, 0)
    
    print(f"\nВсего точек: {n}")
    print(f"Кластеров: {n_clusters}")
    print(f"Точек шума: {n_noise} ({n_noise / n * 100:.1f}%)")
    if n_clusters > 0:
        print(f"Средний размер кластера: {(n - n_noise) / n_clusters:.1f}")
    
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(paths[idx])
    
    print("\nТоп-10 кластеров по размеру:")
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
    # Исключаем шум из топа
    sorted_clusters = [(k, v) for k, v in sorted_clusters if k != -1]
    
    for label, members in sorted_clusters[:10]:
        print(f"  Кластер {label} ({len(members)} файлов):")
        for p in members[:10]:
            print(f"    {p}")
    
    return clusters


def _auto_name_clusters_via_logits(clusters, paths, max_samples_per_cluster=5):
    """Автоматическое именование кластеров через logits_per_image.
    
    Использует SigLIP2 для определения категории по сэмплам из кластера.
    Загружает модель один раз и переиспользует её для всех кластеров.
    """
    if not clusters:
        return {}
    
    try:
        from core.embedder import get_embedder
        embedder = get_embedder()
    except Exception as e:
        print(f"Ошибка загрузки модели для автоименования: {e}")
        return {}
    
    # Английские названия категорий для модели
    english_categories = [CATEGORY_ENGLISH.get(c, c) for c in PREDEFINED_CATEGORIES]
    
    cluster_names = {}
    used_names = {}
    
    for cluster_id, members in sorted(clusters.items()):
        if cluster_id == -1:
            cluster_names[cluster_id] = "Разное"
            continue
            
        if not members:
            continue
        
        # Берём до max_samples_per_cluster изображений из кластера
        sample_paths = members[:max_samples_per_cluster]
        
        # Загружаем изображения
        images = []
        valid_paths = []
        for p in sample_paths:
            try:
                img = embedder._load_image(p)
                images.append(img)
                valid_paths.append(p)
            except Exception:
                continue
        
        if not images:
            continue
        
        try:
            # Прогоняем изображения через модель с текстами категорий
            inputs = embedder.processor(
                images=images,
                text=english_categories,
                padding="max_length",
                return_tensors="pt",
            ).to(embedder.device)
            
            with torch.no_grad():
                outputs = embedder.model(**inputs)
            
            # logits_per_image: (n_images, n_categories)
            logits = outputs.logits_per_image.cpu().numpy()
            
            # Усредняем логиты по изображениям кластера
            avg_logits = np.mean(logits, axis=0)
            
            # Находим категорию с максимальным средним логитом
            best_idx = int(np.argmax(avg_logits))
            best_category = PREDEFINED_CATEGORIES[best_idx]
            best_logit = float(avg_logits[best_idx])
            
            # Проверяем, что логит достаточно высокий
            sorted_logits = np.sort(avg_logits)[::-1]
            if len(sorted_logits) >= 2:
                margin = sorted_logits[0] - sorted_logits[1]
            else:
                margin = 0
            
            margin_threshold = getattr(config, "AUTO_NAME_MARGIN_THRESHOLD", 0.1)
            
            if margin < margin_threshold:
                cluster_names[cluster_id] = "Разное"
                used_names.setdefault("Разное", 0)
                used_names["Разное"] += 1
                if used_names["Разное"] > 1:
                    cluster_names[cluster_id] = f"Разное {used_names['Разное']}"
                continue
            
            if best_category in used_names:
                used_names[best_category] += 1
                cluster_names[cluster_id] = f"{best_category} {used_names[best_category]}"
            else:
                used_names[best_category] = 1
                cluster_names[cluster_id] = best_category
        
        except Exception as e:
            print(f"Ошибка автоименования кластера {cluster_id}: {e}")
            if "Разное" in used_names:
                used_names["Разное"] += 1
                cluster_names[cluster_id] = f"Разное {used_names['Разное']}"
            else:
                used_names["Разное"] = 1
                cluster_names[cluster_id] = "Разное"
    
    return cluster_names


def run(progress_callback=None):
    """Запуск полного пайплайна кластеризации."""
    embeddings, paths = load_embeddings()
    if embeddings is None:
        print("Эмбеддинги не найдены. Сначала выполните: python main.py embed")
        return None
    
    total = len(embeddings)
    if progress_callback is not None:
        progress_callback("cluster", 0, total, "Подготовка к кластеризации HDBSCAN")
    
    print(f"Кластеризация {total} точек через HDBSCAN + UMAP...")
    labels = cluster(embeddings)
    
    if progress_callback is not None:
        progress_callback("cluster", total, total, "Кластеризация завершена")
    
    clusters = print_stats(labels, paths)
    
    # Автоматическое именование кластеров
    if progress_callback is not None:
        progress_callback("cluster", 0, 1, "Автоматическое именование категорий...")
    
    # Автоматическое именование через logits (переиспользуем загруженную модель)
    cluster_names = _auto_name_clusters_via_logits(clusters, paths)
    
    # Сохраняем кластеры с именами
    database.save_clusters(clusters, cluster_names)
    
    if progress_callback is not None:
        progress_callback("cluster", 1, 1, "Категории названы")
    
    print(f"\nКластеры сохранены в БД: {config.DB_FILE}")
    print(f"Автоматически названо категорий: {len(cluster_names)}")
    for cid, name in sorted(cluster_names.items()):
        print(f"  Кластер {cid}: {name}")
    
    return clusters