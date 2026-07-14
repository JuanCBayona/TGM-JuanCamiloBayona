from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

RANDOM_SEED = 42

# Directory holding these scripts (e.g. master/scripts)
SCRIPTS_DIR = Path(__file__).resolve().parent

# One level above the scripts (e.g. master).
# Relative output folders are resolved against this, so that
# --output results  ->  master/results
PROJECT_ROOT = SCRIPTS_DIR.parent

CACHE_DIR = Path("cache")

PCA_COMPONENTS = 50

IMAGE_SIZE = 224

FEATURE_DIM = 1024

EXAMPLE_PAIRS = 8

INCEPTION_IMAGE_SIZE = 299

INCEPTION_FEATURE_DIM = 2048

INCEPTION_NUM_CLASSES = 1000

IS_SPLITS = 10

FID_MIN_SAMPLES = 2048
