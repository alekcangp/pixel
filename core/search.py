"""Поиск изображений: по pHash (LSH) и семантический (SigLIP + FAISS)."""

import sys

import faiss
import numpy as np
import torch

import config
from core.clusterer import load_embeddings
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

    Возвращает все результаты с близостью >= threshold, отсортированные
    по убыванию близости (самый близкий — первый). Если threshold равен None,
    используется config.SEARCH_THRESHOLD. Если threshold == 0.0, возвращаются
    все top_k результатов.
    """
    if top_k is None:
        top_k = config.SEARCH_TOP_K
    if threshold is None:
        threshold = config.SEARCH_THRESHOLD

    embeddings, paths = load_embeddings()
    if embeddings is None:
        print("Эмбеддинги не найдены. Сначала выполните: python main.py embed")
        return []

    print("Поиск: \"%s\"" % query)
    embedder = get_embedder()
    q = embed_query(embedder, query)
    index = build_index(embeddings)

    top_k = min(top_k, len(paths))
    scores, indices = index.search(q, top_k)

    # Фильтруем по порогу близости
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= len(paths):
            continue
        if threshold > 0.0 and score < threshold:
            continue
        results.append((paths[idx], float(score)))

    print("\nНайдено результатов: %d" % len(results))
    for i, (path, score) in enumerate(results):
        print("  %d. %s (близость: %.4f)" % (i + 1, path, score))
    return results
