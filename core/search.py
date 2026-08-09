"""Поиск изображений: по pHash (LSH) и семантический (SigLIP + FAISS)."""

import sys

import faiss
import numpy as np
import torch

import config
from core import database
from core.dedup import LSHIndex, compute_phash
from core.embedder import get_embedder
from core.scanner import load_index


# ============================================================
# Поиск по pHash (LSH)
# ============================================================


def build_hash_index(files):
    threshold = config.PHASH_THRESHOLD

    hashes = []
    paths = []
    total = len(files)
    failed = 0

    for i, f in enumerate(files):
        path = f["path"]
        h = compute_phash(path)
        if h is not None:
            hashes.append(h)
            paths.append(path)
        else:
            failed += 1
        if (i + 1) % 50 == 0 or i + 1 == total:
            pct = (i + 1) * 100 // total
            sys.stdout.write("\rBuilding pHash index: %d/%d (%d%%)" % (i + 1, total, pct))
            sys.stdout.flush()

    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if not hashes:
        return None, [], failed, threshold

    index = LSHIndex(hashes, threshold)

    return index, paths, failed, threshold


def search_by_image(index, query_hash, paths, threshold, top_k=None):
    if index is None or len(index) == 0:
        return []

    if top_k is None:
        top_k = config.SEARCH_TOP_K

    results = index.radius_search(query_hash, radius=threshold)

    return [(dist, paths[idx]) for dist, idx in results[:top_k]]


def run_hash_search(image_path, top_k=None):
    if top_k is None:
        top_k = config.SEARCH_TOP_K

    files = load_index()
    if files is None:
        print("Индекс не найден. Сначала выполните: python main.py scan --path ...")
        return []

    query_hash = compute_phash(image_path)

    if query_hash is None:
        print("Не удалось вычислить хэш для: %s" % image_path)
        return []

    print("Строим LSH-индекс для %d файлов..." % len(files))
    index, paths, failed, threshold = build_hash_index(files)

    if index is None:
        print("Не удалось построить индекс.")
        return []

    print("Индекс построен. Поиск похожих изображений...")
    results = search_by_image(index, query_hash, paths, threshold, top_k=top_k)

    print("\nТоп-%d результатов по pHash (threshold=%d):" % (top_k, threshold))
    if not results:
        print("  Нет совпадений.")
    else:
        for i, (dist, path) in enumerate(results):
            print("  %d. dist=%d  %s" % (i + 1, dist, path))

    print("\nНайдено: %d" % len(results))
    return results


# ============================================================
# Семантический поиск (SigLIP + FAISS)
# ============================================================


# Кэш для индекса и эмбеддингов (чтобы не перестраивать при каждом поиске)
_cache = {
    "embeddings": None,
    "paths": None,
    "index": None,
}


def clear_cache():
    """Сбрасывает кэш эмбеддингов и FAISS-индекса.

    Должен вызываться после пересчёта эмбеддингов (embedder.run), иначе
    текстовый поиск будет возвращать результаты на основе устаревших данных.
    """
    _cache["embeddings"] = None
    _cache["paths"] = None
    _cache["index"] = None


def build_index(embeddings):
    emb = embeddings.astype("float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def embed_query(embedder, text):
    placeholder = np.zeros((config.EMBED_IMAGE_SIZE, config.EMBED_IMAGE_SIZE, 3), dtype=np.uint8)
    inputs = embedder.processor(images=[placeholder], text=text, padding="max_length", return_tensors="pt").to(embedder.device)
    with torch.no_grad():
        outputs = embedder.model(**inputs)
    emb = outputs.text_embeds.cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    return emb


def run(query, top_k=None, threshold=None):
    """Семантический поиск.

    Возвращает результаты, близость которых попадает в интервал
    [best - threshold, best], где best — максимальная близость среди
    найденных. Результаты отсортированы по убыванию близости
    (самый близкий — первый). Если threshold равен None,
    используется config.SEARCH_THRESHOLD. Если threshold == 0.0,
    возвращаются все top_k результатов.
    """
    if top_k is None:
        top_k = config.SEARCH_TOP_K
    if threshold is None:
        threshold = config.SEARCH_THRESHOLD

    # Кэшируем эмбеддинги и индекс, чтобы не перестраивать при каждом поиске
    if _cache["embeddings"] is None or _cache["paths"] is None:
        embeddings, paths = database.load_embeddings()
        if embeddings is None:
            print("Эмбеддинги не найдены. Сначала выполните: python main.py embed")
            return []
        _cache["embeddings"] = embeddings
        _cache["paths"] = paths
    else:
        embeddings = _cache["embeddings"]
        paths = _cache["paths"]

    if _cache["index"] is None:
        print("Построение FAISS-индекса...")
        _cache["index"] = build_index(embeddings)

    print("Поиск: \"%s\"" % query)
    embedder = get_embedder()
    q = embed_query(embedder, query)
    index = _cache["index"]

    top_k = min(top_k, len(paths))
    scores, indices = index.search(q, top_k)

    # Собираем все результаты
    all_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= len(paths):
            continue
        all_results.append((paths[idx], float(score)))

    # Фильтруем по интервалу [best - threshold, best]
    if threshold > 0.0 and all_results:
        best = all_results[0][1]
        min_score = best - threshold
        results = [(p, s) for p, s in all_results if s >= min_score]
    else:
        results = all_results

    print("\nНайдено результатов: %d" % len(results))
    for i, (path, score) in enumerate(results):
        print("  %d. %s (близость: %.4f)" % (i + 1, path, score))
    return results
