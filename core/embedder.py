import logging
import os
import sys
import time
import warnings

import numpy as np
from PIL import Image, ImageFile, ImageOps

# Ограничение потоков OpenMP до 1: PyTorch на Apple Silicon (MPS) при
# инициализации модели в фоновом потоке (asyncio.to_thread) создаёт столько
# OMP-потоков, сколько ядер CPU, что на macOS приводит к
# "OMP: Error #179: Function pthread_mutex_init failed" и segmentation fault
# при запуске GUI. Эту переменную надо выставить ДО импорта torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

ImageFile.LOAD_TRUNCATED_IMAGES = True

import config
from core import database
from core import scanner

try:
    import torch
    from transformers import AutoModel, AutoProcessor
    from transformers.utils import logging as tf_logging
    tf_logging.set_verbosity_error()
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    AutoModel = None
    AutoProcessor = None

# Singleton instance для эмбеддера
_embedder_instance = None


def _detect_device():
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        try:
            free = torch.cuda.mem_get_info()[0]
            if free < 1.5 * 1024**3:
                print("Мало VRAM, fallback на CPU")
                return "cpu"
            return "cuda"
        except Exception:
            return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Используем Apple Silicon MPS")
        return "mps"
    return "cpu"


class SiglipEmbedder:
    def __init__(self):
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch не установлен. Установите torch для эмбеддингов: "
                "pip install torch"
            )
        self.device = _detect_device()
        print(f"Устройство: {self.device}")

        self.processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)
        self.model = AutoModel.from_pretrained(config.SIGLIP_MODEL)

        if self.device == "cuda":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_image(self, path):
        with Image.open(path) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            try:
                img = img.convert("RGB")
            except Exception:
                img = img.convert("RGBA").convert("RGB")
            return img

    def embed_paths(self, paths, batch_size=None, progress_callback=None, batch_save_callback=None):
        if batch_size is None:
            batch_size = config.EMBED_BATCH_SIZE

        embeddings = []
        valid_paths = []
        n = len(paths)
        for i in range(0, n, batch_size):
            if scanner.STOP_REQUESTED:
                print("\nОстановка по запросу пользователя.")
                break
            batch_paths = paths[i:i + batch_size]
            images = []
            batch_valid = []
            for p in batch_paths:
                try:
                    images.append(self._load_image(p))
                    batch_valid.append(p)
                except Exception:
                    continue

            if not images:
                if progress_callback is not None:
                    progress_callback("embed", min(i + batch_size, n), n, "Генерация эмбеддингов")
                continue

            batch_embeddings = []
            batch_embedded_paths = []
            try:
                inputs = self.processor(
                    images=images,
                    text=["image"] * len(images),
                    padding="max_length",
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                emb = outputs.image_embeds.cpu().numpy()
                batch_embeddings.append(emb)
                batch_embedded_paths.extend(batch_valid)
                embeddings.append(emb)
                valid_paths.extend(batch_valid)
            except Exception as e:
                print("\nОшибка обработки батча %d-%d: %s" % (i + 1, min(i + batch_size, n), e))
                for p in batch_valid:
                    try:
                        img = self._load_image(p)
                        inputs = self.processor(
                            images=[img],
                            text=["image"],
                            padding="max_length",
                            return_tensors="pt",
                        ).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        emb = outputs.image_embeds.cpu().numpy()
                        batch_embeddings.append(emb)
                        batch_embedded_paths.append(p)
                        embeddings.append(emb)
                        valid_paths.append(p)
                    except Exception:
                        continue

            if scanner.STOP_REQUESTED:
                print("\nОстановка по запросу пользователя.")
                break

            if batch_embeddings and batch_save_callback is not None:
                try:
                    batch_save_callback(np.vstack(batch_embeddings), batch_embedded_paths)
                except Exception as e:
                    print("\nОшибка сохранения батча в БД: %s" % e)

            if progress_callback is not None:
                progress_callback("embed", min(i + batch_size, n), n, "Генерация эмбеддингов")
            if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= n:
                _progress(min(i + batch_size, n), n, "Эмбеддинги")

        if not embeddings:
            return np.zeros((0, 0)), []
        return np.vstack(embeddings), valid_paths


def _progress(current, total, label="Обработка"):
    pct = current * 100 // total if total else 100
    sys.stdout.write("\r%s: %d/%d (%d%%)" % (label, current, total, pct))
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def get_embedder() -> SiglipEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = SiglipEmbedder()
    return _embedder_instance


def run(progress_callback=None, incremental=True):
    if not _HAS_TORCH:
        print(
            "Эмбеддинги недоступны: PyTorch не установлен. "
            "Установите torch для использования эмбеддингов: "
            "pip install torch"
        )
        return None

    files = scanner.load_index()
    if files is None:
        print("Индекс не найден. Сначала выполните: python main.py scan --path ...")
        return None

    all_paths = [f["path"] for f in files]
    current_mtime_map = {f["path"]: f.get("mtime", 0) for f in files}

    if incremental:
        dup_paths = database.load_duplicate_paths() or set()
        failed_paths = database.load_failed_paths() or set()
        excluded = dup_paths | failed_paths

        existing_embeddings, existing_embed_paths, existing_mtime = database.load_embeddings_with_mtime()
        existing_embed_set = set(existing_embed_paths) if existing_embed_paths else set()

        paths_to_embed = []
        for p in all_paths:
            if p in excluded:
                continue
            if p not in existing_embed_set:
                paths_to_embed.append(p)
            else:
                saved_mtime = existing_mtime.get(p, 0)
                if current_mtime_map.get(p, 0) != saved_mtime:
                    paths_to_embed.append(p)

        print(f"Инкрементальный режим:")
        print(f"  Всего файлов: {len(all_paths)}")
        print(f"  Исключено (дубликаты/повреждённые): {len(excluded)}")
        print(f"  Уже есть эмбеддинги: {len(existing_embed_set)}")
        print(f"  Нужно вычислить: {len(paths_to_embed)}")
    else:
        paths_to_embed = all_paths[:]
        print("Полный режим: пересчёт всех эмбеддингов")
        print(f"  Всего файлов для обработки: {len(paths_to_embed)}")

    if not paths_to_embed:
        print("Нет файлов для обработки.")
        return None

    t0 = time.time()
    embedder = get_embedder()
    if embedder.device == "cuda":
        print(f"Модель готова (CUDA), время инициализации: {time.time() - t0:.1f} с")
    else:
        print(f"Модель готова (CPU), время инициализации: {time.time() - t0:.1f} с")

    if not incremental:
        database.save_embeddings(np.zeros((0, 0)), [], {}, incremental=False)

    def batch_save_callback(batch_vectors, batch_paths):
        batch_mtime = {p: current_mtime_map.get(p, 0) for p in batch_paths}
        database.save_embeddings(batch_vectors, batch_paths, batch_mtime, incremental=True)

    print(f"Эмбеддинг {len(paths_to_embed)} изображений...")
    t0 = time.time()
    embeddings, valid_paths = embedder.embed_paths(
        paths_to_embed,
        progress_callback=progress_callback,
        batch_save_callback=batch_save_callback,
    )
    elapsed = time.time() - t0

    if scanner.STOP_REQUESTED:
        print("\nВычисление эмбеддингов остановлено. Уже обработанные батчи сохранены в БД.")
        return None

    if len(valid_paths) != len(paths_to_embed):
        print(f"Пропущено повреждённых файлов: {len(paths_to_embed) - len(valid_paths)}")

    new_mtime_map = {p: current_mtime_map.get(p, 0) for p in valid_paths}

    if incremental and existing_embeddings is not None and existing_embed_paths:
        all_embeddings = []
        all_paths_ordered = []

        all_embeddings.append(existing_embeddings)
        all_paths_ordered.extend(existing_embed_paths)

        path_to_emb = {p: emb for p, emb in zip(valid_paths, embeddings)}
        final_paths = []
        final_embs = []
        seen = set()
        for p, emb in zip(all_paths_ordered, all_embeddings[0]):
            if p in path_to_emb:
                final_paths.append(p)
                final_embs.append(path_to_emb[p])
                seen.add(p)
            else:
                final_paths.append(p)
                final_embs.append(emb)

        for p, emb in zip(valid_paths, embeddings):
            if p not in seen:
                final_paths.append(p)
                final_embs.append(emb)

        final_embeddings = np.vstack(final_embs) if final_embs else np.zeros((0, 0))
        final_mtime = dict(existing_mtime) if existing_mtime else {}
        final_mtime.update(new_mtime_map)
        database.save_embeddings(final_embeddings, final_paths, final_mtime, incremental=True)
    else:
        database.save_embeddings(embeddings, valid_paths, new_mtime_map, incremental=False)

    print(f"\nОбработано изображений: {len(valid_paths)}")
    if embeddings.shape[0] > 0:
        print(f"Размерность эмбеддингов: {embeddings.shape}")
    print(f"Время: {elapsed:.1f} с")
    print(f"Устройство: {embedder.device}")
    print("\nТоп-10 путей:")
    for p in valid_paths[:10]:
        print(f"  {p}")

    print(f"\nЭмбеддинги сохранены в БД: {config.DB_FILE}")

    try:
        from core import search
        search.clear_cache()
    except Exception:
        pass

    return embeddings, valid_paths
