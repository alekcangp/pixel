import os

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# База данных SQLite (вместо отдельных JSON/npy-файлов)
DB_FILE = os.path.join(STORAGE_DIR, "images.db")

# Сканирование
DEFAULT_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"]

# Минимальный размер файла в байтах (файлы меньше этого размера пропускаются).
MIN_FILE_SIZE = 10 * 1024  # 10 КБ

# Директории, исключаемые по умолчанию (тяжёлые/системные)
DEFAULT_EXCLUDE_DIRS = [
    "AppData", "node_modules", ".git", ".cache", "__pycache__",
    "venv", ".venv", "env", ".env", "site-packages", "dist", "build",
    ".gradle", ".m2", ".npm", ".yarn", ".cargo", ".rustup",
    "Program Files", "ProgramData", "Program Files (x86)", "Windows", "System32",
    "Temp", "tmp", "Trash", ".Trash", "Library", "Application Support", "Games",
    # Python-окружения и пакетные менеджеры
    #"anaconda3", "miniconda3", "conda", "pkgs", "Lib", "Scripts",
    #"site-packages", "python", "Python311", "Python312",
]

# Дедупликация
# pHash (Perceptual Hash) через OpenCV (cv2.img_hash).
# Использует DCT для вычисления эмбеддинга низкочастотных коэффициентов.
# Устойчив к глобальным изменениям яркости и контраста, сжатию с потерями.
#
# Порог Hamming distance для "похожих" изображений.
# Кластеризация выполняется через LSH-подход (разбиение 64-битного хэша
# на подблоки + инвертированный индекс + Union-Find): O(n) построение индекса,
# проверка только кандидатов с совпадающим подблоком (векторизованно через numpy).
PHASH_THRESHOLD = 2            # порог Hamming distance для pHash


# Эмбеддинги
SIGLIP_MODEL = "google/siglip2-base-patch16-224"
EMBED_BATCH_SIZE = 16
EMBED_IMAGE_SIZE = 224

# Zero-shot классификация CLIP
CLIP_MODEL = "openai/clip-vit-base-patch32"
ZERO_SHOT_OUTPUT_DIR = os.path.join(STORAGE_DIR, "classified")
ZERO_SHOT_TEXT_PREFIX = "фото "
ZERO_SHOT_DEFAULT_THRESHOLD = 0.25


# Кластеризация
# CLUSTER_MAX_CLUSTERS больше не используется как жёсткий потолок.
# Верхняя граница определяется динамически: sqrt(n) * 2.
CLUSTER_MAX_CLUSTERS = 10
CLUSTER_MIN_CLUSTERS = 2
CLUSTER_RANDOM_STATE = 42

# Zero-shot classification by embeddings: minimum cosine-similarity margin
# between best and second-best category for a confident assignment.
# Images below this margin are assigned to "Разное".
CLASSIFY_MARGIN_THRESHOLD = 0.001

# Auto-naming clusters: minimum cosine-similarity margin
# between best and second-best category for confident assignment.
AUTO_NAME_MARGIN_THRESHOLD = 0.1

# Auto-naming clusters: minimum absolute score (best category confidence)
# to consider assignment as non-random.
AUTO_NAME_MIN_SCORE = 0.05


# UMAP параметры для снижения размерности
UMAP_N_COMPONENTS = 15         # Целевая размерность после UMAP
UMAP_MIN_DIST = 0.1            # Минимальное расстояние между точками
UMAP_METRIC = "cosine"         # Метрика для UMAP

# HDBSCAN параметры
HDBSCAN_MIN_CLUSTER_SIZE = 5   # Минимальный размер кластера (динамический)
HDBSCAN_MIN_SAMPLES = 3        # Минимальное количество samples в окрестности
HDBSCAN_METRIC = "euclidean"   # Метрика (используем euclidean после UMAP)
HDBSCAN_SELECTION_METHOD = "eom"  # Excess of Mass для переменной плотности


# Поиск
SEARCH_TOP_K = 10
# Порог близости (cosine similarity) для семантического поиска.
# Результаты с близостью ниже этого порога отбрасываются.
# Значение 0.0 означает "показывать все результаты".
SEARCH_THRESHOLD = 0.0