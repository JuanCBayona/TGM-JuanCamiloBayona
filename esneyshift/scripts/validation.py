from pathlib import Path

from io_utils import get_relative_image_map


SUPPORTED_MODEL_TYPES = {

    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",

    "densenet121",
    "densenet169",
    "densenet201",

    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4"
}


def validate_directory(
    path,
    name
):

    if path is None:
        return

    path = Path(path)

    if not path.exists():

        raise RuntimeError(
            f"{name} does not exist:\n{path}"
        )

    if not path.is_dir():

        raise RuntimeError(
            f"{name} is not a directory:\n{path}"
        )


def validate_model_selection(
    checkpoint_path,
    model_type
):

    if (
        checkpoint_path is not None
        and model_type is not None
    ):

        raise RuntimeError(
            "Specify either "
            "--checkpoint or "
            "--model-type, "
            "not both."
        )

    if (
        checkpoint_path is None
        and model_type is None
    ):

        raise RuntimeError(
            "You must specify "
            "either --checkpoint "
            "or --model-type."
        )

    if checkpoint_path is not None:

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise RuntimeError(
                f"Checkpoint not found:\n"
                f"{checkpoint_path}"
            )

    if model_type is not None:

        if model_type not in SUPPORTED_MODEL_TYPES:

            supported = "\n".join(
                sorted(
                    SUPPORTED_MODEL_TYPES
                )
            )

            raise RuntimeError(
                f"Unsupported model type:\n"
                f"{model_type}\n\n"
                f"Supported models:\n"
                f"{supported}"
            )


def validate_conditional_pairs(
    originals_dir,
    objective_dir
):

    originals = get_relative_image_map(
        originals_dir
    )

    objective = get_relative_image_map(
        objective_dir
    )

    missing = []

    for name in originals:

        if name not in objective:

            missing.append(
                name
            )

    if len(missing):

        preview = "\n".join(
            missing[:20]
        )

        raise RuntimeError(
            "Missing generated files:\n"
            f"{preview}"
        )

    if len(originals) != len(objective):

        raise RuntimeError(
            "Originals and objective "
            "contain different numbers "
            "of images."
        )


def validate_all(
    source_dir,
    objective_dir,
    checkpoint_path=None,
    model_type=None,
    originals_dir=None
):

    validate_directory(
        source_dir,
        "Source dataset"
    )

    validate_directory(
        objective_dir,
        "Objective dataset"
    )

    validate_model_selection(
        checkpoint_path,
        model_type
    )

    if originals_dir is not None:

        validate_directory(
            originals_dir,
            "Originals dataset"
        )

        validate_conditional_pairs(
            originals_dir,
            objective_dir
        )
