"""Поиск изображений: по pHash (LSH) и семантический (SigLIP + FAISS)."""

import sys

import faiss
import numpy as np

import config
from core import database
from core.dedup import LSHIndex, compute_phash
from core.embedder import get_embedder, _HAS_TORCH as _EMBEDDER_HAS_TORCH
from core.scanner import load_index

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None


# ============================================================
# FAISS-индекс (строится в памяти, без дискового кэша)
# ============================================================

_cache = {
    "embeddings": None,
    "paths": None,
    "index": None,
}


def build_index(embeddings):
    """Строит нормализованный FAISS-индекс для косинусного сходства.

    Векторы L2-нормализуются здесь (на лету, в памяти), т.к. в БД хранятся
    исходные векторы. IndexFlatIP + normalize_L2 = косинусное сходство.
    """
    emb = embeddings.astype("float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def get_or_build_index(embeddings, paths):
    """Возвращает индекс, строя его в памяти и кэшируя.

    Выполняется быстро (нормализация + add = O(N·D)); индекс строится один раз
    за процесс и переиспользуется через _cache. Дисковый кэш не используется
    намеренно — построение дешёвое, а фоновый prefetch_index скрывает задержку.
    """
    if _cache["index"] is None:
        _cache["index"] = build_index(embeddings)
    return _cache["index"]


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


def clear_cache():
    """Сбрасывает кэш эмбеддингов и FAISS-индекса (в памяти)."""
    _cache["embeddings"] = None
    _cache["paths"] = None
    _cache["index"] = None


def _load_embeddings_sync():
    """Загружает эмбеддинги из БД с кэшированием в памяти.

    Returns:
        (embeddings, paths) или (None, None), если эмбеддингов нет.
    """
    if _cache["embeddings"] is None or _cache["paths"] is None:
        embeddings, paths = database.load_embeddings()
        if embeddings is None:
            return None, None
        _cache["embeddings"] = embeddings
        _cache["paths"] = paths
    return _cache["embeddings"], _cache["paths"]


def prefetch_index():
    """Предзагружает эмбеддинги и FAISS-индекс в фоне (вызывается при старте).

    Заполняет _cache эмбеддингами и готовит индекс (из дискового кэша или
    перестраивая его), чтобы первый поиск выполнялся мгновенно.
    """
    embeddings, paths = _load_embeddings_sync()
    if embeddings is not None:
        get_or_build_index(embeddings, paths)


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

    Фильтруем СРАЗУ по порогу через range_search: сначала узнаём близость
    лучшего результата, вычисляем min_score = best - threshold и запрашиваем
    у FAISS только тех, чья близость реально выше порога. Так кандидаты
    никогда не "закончатся раньше", чем сработает порог.
    """
    if threshold is None:
        threshold = config.SEARCH_THRESHOLD

    if not _HAS_TORCH or not _EMBEDDER_HAS_TORCH:
        print("Семантический поиск недоступен: PyTorch не установлен.")
        print("Установите torch для использования семантического поиска: pip install torch")
        return []

    embeddings, paths = _load_embeddings_sync()
    if embeddings is None:
        print("Эмбеддинги не найдены. Сначала выполните: python main.py embed")
        return []

    # Индекс строится в памяти один раз и кэшируется (см. get_or_build_index);
    # prefetch_index предзагружает его в фоне при старте приложения.
    index = get_or_build_index(embeddings, paths)

    print("Поиск: \"%s\"" % query)
    embedder = get_embedder()
    q = embed_query(embedder, query)

    # Узнаём близость лучшего результата, чтобы вычислить абсолютный порог.
    best = 0.0
    if len(paths) > 0:
        best_scores, _ = index.search(q, 1)
        best = float(best_scores[0][0]) if best_scores.size > 0 else 0.0

    if threshold > 0.0 and best > 0.0:
        # Минимальная близость для попадания в результат.
        min_score = best - threshold
        # range_search возвращает ВСЕ векторы с близостью >= min_score
        # (не ограничиваясь заранее заданным top_k кандидатов).
        lims, D, I = index.range_search(q, min_score)
        start, end = int(lims[0]), int(lims[1])

        pairs = []
        for j in range(start, end):
            idx = int(I[j])
            if idx < len(paths):
                pairs.append((float(D[j]), idx))

        # Сортируем по убыванию близости (самое похожее первым).
        pairs.sort(key=lambda x: x[0], reverse=True)

        all_results = [(paths[idx], score) for score, idx in pairs]
    else:
        # Порог не задан/не сработал — возвращаем всех (или top_k как потолок).
        if top_k is None or top_k <= 0:
            top_k = len(paths)
        top_k = min(top_k, len(paths))
        scores, indices = index.search(q, top_k)
        all_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(paths):
                all_results.append((paths[idx], float(score)))

    # top_k — защитный потолок на случай слишком мягкого порога
    # (например, threshold=0, когда все изображения прошли бы порог).
    if top_k is not None and top_k > 0:
        all_results = all_results[:top_k]

    results = all_results

    print("\nНайдено результатов: %d" % len(results))
    for i, (path, score) in enumerate(results):
        print("  %d. %s (близость: %.4f)" % (i + 1, path, score))
    return results
