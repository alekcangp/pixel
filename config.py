import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

# SQLite database
DB_FILE = os.path.join(STORAGE_DIR, "images.db")

# Scanning
DEFAULT_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"]

# Minimum file size in bytes.
# Files smaller than this value are skipped.
MIN_FILE_SIZE = 20 * 1024  # 20 KiB

# Directories excluded by default.
# These are typically system, cache, development, or dependency directories.
DEFAULT_EXCLUDE_DIRS = [
    # Windows system directories
    "Windows",
    "System32",
    "SysWOW64",
    "Program Files",
    "Program Files (x86)",
    "ProgramData",
    "AppData",
    "Config.Msi",
    "$Recycle.Bin",
    "System Volume Information",
    "WUTemp",
    "WindowsUpdate",
    "$WinREAgent",
    "$WINDOWS.~BT",
    "$WINDOWS.~WS",

    # macOS system directories
    "System",
    "Library",
    "Applications",
    "Volumes",
    "Network",
    "cores",
    "private",
    "usr",
    "bin",
    "sbin",
    "Mobile Documents",
    "Photos Library.photoslibrary",

    # Temporary files and trash
    "Temp",
    "tmp",
    ".cache",
    "Trash",
    ".Trash",
    ".Trashes",

    # Development and build directories
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "bower_components",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
    "site-packages",
    ".gradle",
    ".m2",
    ".npm",
    ".yarn",
    ".cargo",
    ".rustup",
    "dist",
    "build",
    "out",
    "target",
    ".idea",
    ".vscode",
]

# Perceptual deduplication
# Maximum Hamming distance between pHash values considered similar.
PHASH_THRESHOLD = 3

# Image embeddings
SIGLIP_MODEL = "google/siglip-base-patch16-224"
EMBED_BATCH_SIZE = 16
EMBED_IMAGE_SIZE = 224

# Clustering
CLUSTER_MIN_CLUSTERS = 2
CLUSTER_RANDOM_STATE = 42

# UMAP parameters
UMAP_N_COMPONENTS = 3
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_N_NEIGHBORS = 20

# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_SELECTION_METHOD = "eom"

# Automatically run clustering after scanning/embedding.
AUTO_CLUSTER_AFTER_SCAN = True

# ============================================================
# Level 2: Mega-cluster refinement
# ============================================================

MEGA_CLUSTER_THRESHOLD = 0.07

# Local UMAP projection for detecting micro-structures.
SUB_UMAP_N_NEIGHBORS = 10
SUB_UMAP_N_COMPONENTS = 2
SUB_UMAP_MIN_DIST = 0.0
SUB_UMAP_METRIC = "cosine"

# Sub-HDBSCAN: extract dense local structures.
SUB_HDBSCAN_MIN_CLUSTER_SIZE = 20
SUB_HDBSCAN_MIN_SAMPLES = 5
SUB_HDBSCAN_SELECTION_METHOD = "leaf"

# ============================================================
# KNN rescue
# ============================================================

KNN_N_NEIGHBORS = 5
KNN_METRIC = "euclidean"

# Label used for confirmed visual garbage.
# This is different from the HDBSCAN noise label (-1).
GARBAGE_LABEL = -2

# ============================================================
# Noise rescue
# ============================================================

# If more than this fraction of points are classified as noise,
# the KNN-based noise rescue step is enabled.
NOISE_RESCUE_THRESHOLD = 0.15

# Maximum distance in UMAP space for assigning a noise point
# to a nearby cluster.
NOISE_RESCUE_MAX_DIST = 0.5

# Semantic search
SEARCH_TOP_K = 10
SEARCH_THRESHOLD = 0.05