from pathlib import Path

from config import SUPPORTED_EXTENSIONS


def ensure_dir(path):

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def get_image_paths(folder):

    folder = Path(folder)

    image_paths = []

    for ext in SUPPORTED_EXTENSIONS:

        image_paths.extend(
            folder.rglob(f"*{ext}")
        )

        image_paths.extend(
            folder.rglob(f"*{ext.upper()}")
        )

    image_paths = sorted(image_paths)

    return image_paths


def dataset_name_from_path(path):

    return Path(path).name
