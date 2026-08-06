from collections import Counter

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

import config
from core import database


# Предопределенные категории для автоименования
PREDEFINED_CATEGORIES = [
    "Природа", "Город", "Люди", "Животные", "Еда",
    "Архитектура", "Пейзаж", "Портрет", "Улица", "Интерьер",
    "Техника", "Спорт", "Искусство", "Транспорт", "Цветы",
    "Море", "Горы", "Лес", "Небо", "Дорога"
]

# Английские эквиваленты для эмбеддингов (модель CLIP/SigLIP лучше работает с английским)
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
    """L2-нормализация эмбеддингов.

    KMeans использует евклидово расстояние. После L2-нормализации
    евклидово расстояние монотонно связано с косинусным, что улучшает
    качество кластеризации семантических эмбеддингов.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _select_optimal_clusters(embeddings, progress_callback=None):
    """Выбор оптимального числа кластеров через комбинированную метрику.

    Использует нормализованную комбинацию двух метрик:
      - Silhouette Score (максимум — лучше): оценивает, насколько точки
        ближе к своему кластеру, чем к соседнему. Хорошо улавливает
        семантическую разделимость.
      - Davies-Bouldin Index (минимум — лучше): отношение внутрикластерного
        разброса к межкластерному расстоянию. Компактность vs раздельность.

    Комбинация обеих метрик (нормализованных к [0,1]) даёт более устойчивый
    выбор, чем каждая по отдельности: silhouette предотвращает выбор
    слишком малого числа кластеров, а DBI — слишком большого.
    """
    n_samples = len(embeddings)
    if n_samples < 2:
        return 1

    # Динамический диапазон: масштабируется с размером датасета
    # sqrt(n) * 2 — эвристика из эмпирического правила для k-means
    # Верхняя граница определяется автоматически, без жёсткого потолка
    dynamic_max = int(np.sqrt(n_samples) * 2)
    max_clusters = min(dynamic_max, n_samples - 1)
    min_clusters = min(config.CLUSTER_MIN_CLUSTERS, max_clusters)
    if max_clusters < 2:
        return 1

    sil_scores = {}
    dbi_scores = {}

    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=config.CLUSTER_RANDOM_STATE,
                batch_size=256,
            )
            labels = km.fit_predict(embeddings)
            sil_scores[n_clusters] = silhouette_score(embeddings, labels)
            dbi_scores[n_clusters] = davies_bouldin_score(embeddings, labels)
        except Exception:
            continue

        if progress_callback is not None:
            progress_callback(
                "cluster", n_clusters - min_clusters,
                max_clusters - min_clusters,
                "Подбор оптимального числа кластеров"
            )

    if not sil_scores:
        return min_clusters

    # Нормализация метрик к [0, 1]
    sil_vals = np.array(list(sil_scores.values()))
    dbi_vals = np.array(list(dbi_scores.values()))

    sil_range = sil_vals.max() - sil_vals.min()
    dbi_range = dbi_vals.max() - dbi_vals.min()

    ks = list(sil_scores.keys())
    combined_scores = {}
    for i, k in enumerate(ks):
        sil_n = (sil_vals[i] - sil_vals.min()) / (sil_range + 1e-10)
        dbi_n = (dbi_vals[i] - dbi_vals.min()) / (dbi_range + 1e-10)
        # Silhouette: больше — лучше; DBI: меньше — лучше
        combined_scores[k] = sil_n + (1.0 - dbi_n)

    best_n = max(combined_scores, key=combined_scores.get)
    return best_n


def cluster(embeddings, n_clusters=None):
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)

    # L2-нормализация для корректной работы KMeans с семантическими эмбеддингами
    embeddings = _l2_normalize(embeddings)

    if n_clusters is None:
        n_clusters = _select_optimal_clusters(embeddings)
    n_clusters = min(max(1, n_clusters), len(embeddings))
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.CLUSTER_RANDOM_STATE,
        batch_size=256,
    )
    labels = km.fit_predict(embeddings)
    return labels


def print_stats(labels, paths):
    n = len(labels)
    counter = Counter(labels)
    print(f"Всего точек: {n}")
    print(f"Кластеров: {len(counter)}")
    print(f"Средний размер кластера: {n / len(counter):.1f}")

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(paths[idx])

    print("\nТоп-10 кластеров по размеру:")
    for label, members in sorted(clusters.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  Кластер {label} ({len(members)} файлов):")
        for p in members[:3]:
            print(f"    {p}")

    return clusters


def _auto_name_clusters_via_logits(clusters, paths, max_samples_per_cluster=5):
    """Автоматическое именование кластеров через logits_per_image.

    SigLIP2 — мультимодальная модель. image_embeds и text_embeds находятся
    в разных пространствах, поэтому косинусное сходство между ними не работает.
    Вместо этого прогоняем изображения через модель вместе с текстами категорий
    и используем logits_per_image для определения наиболее подходящей категории.

    Args:
        clusters: dict {cluster_id: [path, ...]}
        paths: list[str] — все пути с эмбеддингами
        max_samples_per_cluster: int — сколько изображений брать из кластера

    Returns:
        dict {cluster_id: name}
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

    for cluster_id, members in clusters.items():
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

            # Проверяем, что логит достаточно высокий (не все категории одинаково плохие)
            # Порог: разница между лучшей и средней категорией
            sorted_logits = np.sort(avg_logits)[::-1]
            if len(sorted_logits) >= 2:
                margin = sorted_logits[0] - sorted_logits[1]
            else:
                margin = 0

            # Если разница между лучшей и второй категорией мала, используем "Разное"
            if margin < 0.5:
                # Проверяем, не использовано ли уже имя "Разное"
                if "Разное" in used_names:
                    used_names["Разное"] += 1
                    cluster_names[cluster_id] = f"Разное {used_names['Разное']}"
                else:
                    used_names["Разное"] = 1
                    cluster_names[cluster_id] = "Разное"
                continue

            # Проверяем, не использовано ли уже это имя
            if best_category in used_names:
                used_names[best_category] += 1
                cluster_names[cluster_id] = f"{best_category} {used_names[best_category]}"
            else:
                used_names[best_category] = 1
                cluster_names[cluster_id] = best_category

        except Exception as e:
            print(f"Ошибка автоименования кластера {cluster_id}: {e}")
            # Проверяем, не использовано ли уже имя "Разное"
            if "Разное" in used_names:
                used_names["Разное"] += 1
                cluster_names[cluster_id] = f"Разное {used_names['Разное']}"
            else:
                used_names["Разное"] = 1
                cluster_names[cluster_id] = "Разное"

    return cluster_names


def run(progress_callback=None):
    embeddings, paths = load_embeddings()
    if embeddings is None:
        print("Эмбеддинги не найдены. Сначала выполните: python main.py embed")
        return None

    total = len(embeddings)
    if progress_callback is not None:
        progress_callback("cluster", 0, total, "Подготовка к кластеризации")
    print(f"Кластеризация {total} точек...")
    labels = cluster(embeddings)
    if progress_callback is not None:
        progress_callback("cluster", total, total, "Кластеризация завершена")
    clusters = print_stats(labels, paths)

    # Автоматическое именование кластеров
    if progress_callback is not None:
        progress_callback("cluster", 0, 1, "Автоматическое именование категорий...")

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