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

CACHE_DIR = Path("cache")

PCA_COMPONENTS = 50

IMAGE_SIZE = 224

FEATURE_DIM = 1024

EXAMPLE_PAIRS = 8
