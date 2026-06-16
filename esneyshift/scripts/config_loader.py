from pathlib import Path

import yaml


def load_config(config_file):

    config_file = Path(config_file)

    if not config_file.exists():

        raise RuntimeError(
            f"Config file not found:\n"
            f"{config_file}"
        )

    with open(
        config_file,
        "r"
    ) as f:

        config = yaml.safe_load(f)

    if config is None:

        config = {}

    return config


def merge_config_with_args(
    args,
    config
):

    for key, value in config.items():

        current_value = getattr(
            args,
            key,
            None
        )

        if (
            current_value is None
            or
            (
                isinstance(
                    current_value,
                    str
                )
                and
                current_value == ""
            )
        ):

            setattr(
                args,
                key,
                value
            )

    return args
