import os

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

# База данных SQLite (вместо отдельных JSON/npy-файлов)
DB_FILE = os.path.join(STORAGE_DIR, "images.db")

# Сканирование
DEFAULT_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"]

# Минимальный размер файла в байтах (файлы меньше этого размера пропускаются).
MIN_FILE_SIZE = 20 * 1024  # 10 КБ

# Директории, исключаемые по умолчанию (тяжёлые/системные)
DEFAULT_EXCLUDE_DIRS = [
    "AppData", "node_modules", ".git", ".cache", "__pycache__",
    "venv", ".venv", "env", ".env", "site-packages", "dist", "build",
    ".gradle", ".m2", ".npm", ".yarn", ".cargo", ".rustup",
    "Program Files", "ProgramData", "Program Files (x86)", "Windows", "System32",
    "Temp", "tmp", "Trash", ".Trash", "Library", "Application Support", "Games",
    # macOS служебные папки
    ".DS_Store", ".Spotlight-V100", ".fseventsd", ".DocumentRevisions-V100",
    ".TemporaryItems", ".VolumeIcon.icns", ".com.apple.timemachine.donotpresent",
    "Photos Library.photoslibrary", "Mobile Documents", ".Trashes",
]

# Дедупликация
PHASH_THRESHOLD = 3            # порог Hamming distance для pHash


# Эмбеддинги
SIGLIP_MODEL = "google/siglip2-base-patch16-224"
EMBED_BATCH_SIZE = 16
EMBED_IMAGE_SIZE = 224

# Кластеризация
CLUSTER_MIN_CLUSTERS = 2
CLUSTER_RANDOM_STATE = 42


# UMAP параметры для снижения размерности
UMAP_N_COMPONENTS = 3          # Целевая размерность после UMAP
UMAP_MIN_DIST = 0.0            # Минимальное расстояние между точками
UMAP_METRIC = "cosine"         # Метрика для UMAP
UMAP_N_NEIGHBORS = 20          # Увеличено с 15: более глобальная структура, меньше шума

# HDBSCAN параметры
HDBSCAN_MIN_CLUSTER_SIZE = 15   # Уменьшено с 25: больше мелких кластеров, меньше шума
HDBSCAN_MIN_SAMPLES = 3         # Уменьшено с 5: менее строгий порог плотности
HDBSCAN_SELECTION_METHOD = "eom"  # Excess of Mass для переменной плотности

# Автоматическая кластеризация после сканирования
AUTO_CLUSTER_AFTER_SCAN = True


# ============================================================
# Level 2: Взрыв мега-кластера (иерархическая кластеризация)
# ============================================================
MEGA_CLUSTER_THRESHOLD = 0.07

# Sub-UMAP: жёсткая локальная проекция для выявления микро-структур
SUB_UMAP_N_NEIGHBORS = 10       # Локальное фракционирование
SUB_UMAP_N_COMPONENTS = 2       # Строгая топологическая проекция (плотность)
SUB_UMAP_MIN_DIST = 0.0
SUB_UMAP_METRIC = "cosine"

# Sub-HDBSCAN: извлечение только острых пиков плотности
SUB_HDBSCAN_MIN_CLUSTER_SIZE = 20  # Верхняя граница, адаптируется вниз
SUB_HDBSCAN_MIN_SAMPLES = 5
SUB_HDBSCAN_SELECTION_METHOD = "leaf"  # Срезает пики, плоский фон → шум -1

# ============================================================
# KNN Rescue: переклассификация спасённых точек
# ============================================================
KNN_N_NEIGHBORS = 5
KNN_METRIC = "euclidean"

# Метка для подтверждённого визуального мусора (не путать с шумом -1)
GARBAGE_LABEL = -2

# ============================================================
# Noise Rescue: присоединение шумовых точек к ближайшим кластерам
# ============================================================
# Порог (доля от всех точек), при котором шум считается избыточным
# и запускается пост-обработка шума через KNN.
NOISE_RESCUE_THRESHOLD = 0.15   # Если шум > 15% — запускаем rescue
# Максимальное расстояние (в UMAP-пространстве) для присоединения шумовой
# точки к кластеру. Точки дальше этого порога остаются шумом.
NOISE_RESCUE_MAX_DIST = 0.5


# Поиск
SEARCH_TOP_K = 10
SEARCH_THRESHOLD = 0.05