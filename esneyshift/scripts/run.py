import argparse
import hashlib
import json

from pathlib import Path

import pandas as pd
import torch

from extract_features import (
    extract_embeddings,
    save_features,
    load_features
)

from metrics import Metrics

from image_metrics import (
    ImageMetrics
)

from visualization import (
    Visualizer
)

from validation import (
    validate_all
)

from report import (
    ReportGenerator
)

from io_utils import (
    ensure_dir,
    save_json,
    get_image_paths
)

from config import CACHE_DIR


def get_feature_file(
    dataset_path,
    checkpoint_path,
    model_type
):

    cache_key = (
        str(dataset_path)
        +
        str(checkpoint_path)
        +
        str(model_type)
    )

    path_hash = hashlib.md5(
        cache_key.encode()
    ).hexdigest()[:12]

    return (
        CACHE_DIR /
        f"{path_hash}.npy"
    )


def get_metadata_file(
    dataset_path,
    checkpoint_path,
    model_type
):

    cache_key = (
        str(dataset_path)
        +
        str(checkpoint_path)
        +
        str(model_type)
    )

    path_hash = hashlib.md5(
        cache_key.encode()
    ).hexdigest()[:12]

    return (
        CACHE_DIR /
        f"{path_hash}.json"
    )


def get_feature_extractor_name(
    checkpoint_path,
    model_type
):

    if checkpoint_path is not None:
        return "uni"

    return model_type


def load_or_extract(
    dataset_path,
    checkpoint_path,
    model_type,
    batch_size,
    num_workers,
    device
):

    feature_file = get_feature_file(
        dataset_path,
        checkpoint_path,
        model_type
    )

    metadata_file = get_metadata_file(
        dataset_path,
        checkpoint_path,
        model_type
    )

    if feature_file.exists():

        print(
            "\nLoading cached features:"
        )

        print(
            feature_file
        )

        features = load_features(
            feature_file
        )

        print(
            f"Loaded shape: "
            f"{features.shape}"
        )

        return features

    print(
        "\nExtracting features from:"
    )

    print(
        dataset_path
    )

    features = extract_embeddings(
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )

    save_features(
        features,
        feature_file
    )

    save_json(
        {
            "dataset":
            str(dataset_path),

            "feature_extractor":
            get_feature_extractor_name(
                checkpoint_path,
                model_type
            ),

            "checkpoint":
            (
                None
                if checkpoint_path is None
                else str(
                    checkpoint_path
                )
            ),

            "model_type":
            model_type,

            "feature_shape":
            list(
                features.shape
            )
        },
        metadata_file
    )

    print(
        "\nSaved cache:"
    )

    print(
        feature_file
    )

    return features


def save_embedding_metrics(
    metrics,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    with open(
        output_dir /
        "metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_dir /
        "metrics.csv",
        index=False
    )


def save_conditional_results(
    results,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    pd.DataFrame(
        results[
            "pair_results"
        ]
    ).to_csv(
        output_dir /
        "conditional_metrics.csv",
        index=False
    )

    with open(
        output_dir /
        "conditional_metrics_summary.json",
        "w"
    ) as f:

        json.dump(
            results[
                "summary"
            ],
            f,
            indent=4
        )


def print_results(
    results,
    title
):

    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)

    for key, value in results.items():

        print(
            f"{key:<25} "
            f": {value:.6f}"
        )

    print("=" * 60)


def save_run_metadata(
    args,
    device,
    output_dir
):

    metadata = {

        "feature_extractor":
        get_feature_extractor_name(
            args.checkpoint,
            args.model_type
        ),

        "checkpoint":
        (
            None
            if args.checkpoint is None
            else str(
                args.checkpoint
            )
        ),

        "model_type":
        args.model_type,

        "source_path":
        str(args.source),

        "objective_path":
        str(args.objective),

        "originals_path":
        (
            None
            if args.originals is None
            else str(
                args.originals
            )
        ),

        "device":
        device,

        "source_images":
        len(
            get_image_paths(
                args.source
            )
        ),

        "objective_images":
        len(
            get_image_paths(
                args.objective
            )
        ),

        "original_images":
        (
            None
            if args.originals is None
            else len(
                get_image_paths(
                    args.originals
                )
            )
        )
    }

    save_json(
        metadata,
        Path(output_dir)
        / "metadata.json"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Dataset comparison "
            "using deep features"
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "UNI checkpoint path"
        )
    )

    parser.add_argument(
        "--model-type",
        default=None,
        help=(
            "resnet18, "
            "resnet34, "
            "resnet50, "
            "resnet101, "
            "densenet121, "
            "densenet169, "
            "densenet201, "
            "efficientnet_b0-b4"
        )
    )

    parser.add_argument(
        "--source",
        required=True
    )

    parser.add_argument(
        "--objective",
        required=True
    )

    parser.add_argument(
        "--originals",
        default=None
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8
    )

    parser.add_argument(
        "--output",
        default="results"
    )

    args = parser.parse_args()

    validate_all(
        source_dir=args.source,
        objective_dir=args.objective,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        originals_dir=args.originals
    )

    ensure_dir(
        CACHE_DIR
    )

    ensure_dir(
        args.output
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: "
        f"{device}"
    )

    source_features = load_or_extract(
        dataset_path=args.source,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )

    objective_features = load_or_extract(
        dataset_path=args.objective,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )

    print(
        "\nSource feature shape:"
    )

    print(
        source_features.shape
    )

    print(
        "\nObjective feature shape:"
    )

    print(
        objective_features.shape
    )

    embedding_metrics = (
        Metrics.evaluate(
            source_features,
            objective_features
        )
    )

    print_results(
        embedding_metrics,
        "EMBEDDING METRICS"
    )

    save_embedding_metrics(
        embedding_metrics,
        args.output
    )

    print(
        "\nCreating embedding visualizations..."
    )

    Visualizer.create_embedding_visualizations(
        source_features,
        objective_features,
        embedding_metrics,
        args.output
    )

    conditional_results = None

    if args.originals:

        print(
            "\nComputing conditional metrics..."
        )

        conditional_results = (
            ImageMetrics.evaluate(
                originals_dir=args.originals,
                objective_dir=args.objective
            )
        )

        print_results(
            conditional_results[
                "summary"
            ],
            "CONDITIONAL METRICS"
        )

        save_conditional_results(
            conditional_results,
            args.output
        )

        print(
            "\nCreating conditional visualizations..."
        )

        Visualizer.create_conditional_visualizations(
            conditional_results,
            args.output
        )

    save_run_metadata(
        args=args,
        device=device,
        output_dir=args.output
    )

    ReportGenerator.create_report(
        output_file=(
            Path(args.output)
            / "report.md"
        ),

        source_path=args.source,

        objective_path=args.objective,

        originals_path=args.originals,

        embedding_metrics=embedding_metrics,

        conditional_metrics=(
            None
            if conditional_results is None
            else conditional_results[
                "summary"
            ]
        )
    )

    print(
        "\nFinished."
    )

    print(
        "\nResults saved in:"
    )

    print(
        Path(
            args.output
        ).resolve()
    )


if __name__ == "__main__":
    main()
