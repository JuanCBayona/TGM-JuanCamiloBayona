from pathlib import Path

from config import (
    SUPPORTED_EXTENSIONS,
    PROJECT_ROOT
)


def resolve_output_dir(output):
    """
    Resolves the results folder relative to the project root
    (one level above the scripts folder), not to the current
    working directory.

        --output results  ->  <project_root>/results

    Absolute paths (and paths starting with ~) are respected
    as given.
    """

    output = Path(
        output
    ).expanduser()

    if output.is_absolute():

        return output

    return (
        PROJECT_ROOT
        /
        output
    ).resolve()


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


def get_relative_image_map(folder):

    folder = Path(folder)

    image_map = {}

    for path in get_image_paths(folder):

        rel = str(
            path.relative_to(folder)
        )

        image_map[rel] = path

    return image_map


def save_json(
    data,
    output_file
):

    import json

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )
