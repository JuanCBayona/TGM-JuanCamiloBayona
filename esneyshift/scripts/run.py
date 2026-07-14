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

from inception_features import (
    load_or_extract_inception
)

from generative_metrics import (
    GenerativeMetrics
)

from image_metrics import (
    ImageMetrics
)

from visualization import (
    Visualizer
)

from validation import (
    validate_all,
    is_self_comparison
)

from report import (
    ReportGenerator
)

from io_utils import (
    ensure_dir,
    save_json,
    get_image_paths,
    resolve_output_dir
)

from config import CACHE_DIR

from config_loader import (
    load_config,
    merge_config_with_args
)


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
        CACHE_DIR
        /
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
        CACHE_DIR
        /
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

    ensure_dir(
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
            f"{key:<25}: {value:.6f}"
        )

    print("=" * 60)


def run_embedding_comparison(
    comparison_name,
    features_a,
    features_b,
    output_dir,
    inception_a=None,
    inception_b=None,
    label_a="source",
    label_b="objective"
):

    print(
        f"\nRunning comparison: "
        f"{comparison_name}"
    )

    comparison_dir = (
        Path(output_dir)
        / comparison_name
    )

    ensure_dir(
        comparison_dir
    )

    metrics = Metrics.evaluate(
        features_a,
        features_b
    )

    if (
        inception_a is not None
        and
        inception_b is not None
    ):

        metrics.update(
            GenerativeMetrics.evaluate(
                source_features=inception_a[0],
                objective_features=inception_b[0],
                source_probabilities=inception_a[1],
                objective_probabilities=inception_b[1],
                source_label=label_a,
                objective_label=label_b
            )
        )

    print_results(
        metrics,
        comparison_name.upper()
    )

    save_embedding_metrics(
        metrics,
        comparison_dir
    )

    print(
        "\nCreating visualizations..."
    )

    Visualizer.create_embedding_visualizations(
        features_a,
        features_b,
        metrics,
        comparison_dir
    )

    return metrics


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

        "fid_is_enabled":
        (
            not args.skip_fid_is
        ),

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
        description="Dataset comparison using deep features"
    )

    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--originals", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default="results")

    parser.add_argument(
        "--allow-self-comparison",
        action="store_true",
        default=None,
        help=(
            "Allow --source and --objective to point at the "
            "same folder. Sanity check: every distance should "
            "come back ~0 and CosineSimilarity ~1.0."
        )
    )

    parser.add_argument(
        "--skip-fid-is",
        action="store_true",
        default=None,
        help=(
            "Skip FID / Inception Score "
            "(avoids the extra InceptionV3 pass "
            "over every dataset)."
        )
    )

    args = parser.parse_args()

    if args.config is not None:

        config = load_config(
            args.config
        )

        args = merge_config_with_args(
            args,
            config
        )

    args.output = resolve_output_dir(
        args.output
    )

    validate_all(
        source_dir=args.source,
        objective_dir=args.objective,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        originals_dir=args.originals,
        allow_self_comparison=bool(
            args.allow_self_comparison
        )
    )

    self_comparison = is_self_comparison(
        args.source,
        args.objective
    )

    if self_comparison:

        print("\n")
        print("=" * 60)
        print("SELF-COMPARISON (SANITY CHECK)")
        print("=" * 60)
        print(
            "Source and objective are the same folder.\n\n"
            "Expected: KL, JS, EMD, MMD, KS, Frechet and FID\n"
            "all ~0, CosineSimilarity ~1.0, and both IS values\n"
            "identical.\n\n"
            "NLL will NOT be 0: it is the cross-entropy of a GMM\n"
            "fitted on the source, so it stays at the entropy of\n"
            "the data itself. Use its value here as the 'no shift'\n"
            "reference point for real comparisons.\n\n"
            "Tiny non-zero values (~1e-6) in Frechet/FID are\n"
            "floating-point error in the matrix square root, not\n"
            "real shift."
        )
        print("=" * 60)

    ensure_dir(CACHE_DIR)
    ensure_dir(args.output)

    print(
        f"\nSaving results to: {args.output}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    source_features = load_or_extract(
        args.source,
        args.checkpoint,
        args.model_type,
        args.batch_size,
        args.num_workers,
        device
    )

    objective_features = load_or_extract(
        args.objective,
        args.checkpoint,
        args.model_type,
        args.batch_size,
        args.num_workers,
        device
    )

    originals_features = None

    if args.originals is not None:

        originals_features = load_or_extract(
            args.originals,
            args.checkpoint,
            args.model_type,
            args.batch_size,
            args.num_workers,
            device
        )

    source_inception = None
    objective_inception = None
    originals_inception = None

    if not args.skip_fid_is:

        print(
            "\nPreparing InceptionV3 activations "
            "for FID / IS..."
        )

        source_inception = load_or_extract_inception(
            args.source,
            args.batch_size,
            args.num_workers,
            device
        )

        objective_inception = load_or_extract_inception(
            args.objective,
            args.batch_size,
            args.num_workers,
            device
        )

        if args.originals is not None:

            originals_inception = (
                load_or_extract_inception(
                    args.originals,
                    args.batch_size,
                    args.num_workers,
                    device
                )
            )

    source_vs_objective_metrics = (
        run_embedding_comparison(
            "source_vs_objective",
            source_features,
            objective_features,
            args.output,
            inception_a=source_inception,
            inception_b=objective_inception,
            label_a="source",
            label_b="objective"
        )
    )

    source_vs_originals_metrics = None
    originals_vs_objective_metrics = None

    if originals_features is not None:

        source_vs_originals_metrics = (
            run_embedding_comparison(
                "source_vs_originals",
                source_features,
                originals_features,
                args.output,
                inception_a=source_inception,
                inception_b=originals_inception,
                label_a="source",
                label_b="originals"
            )
        )

        originals_vs_objective_metrics = (
            run_embedding_comparison(
                "originals_vs_objective",
                originals_features,
                objective_features,
                args.output,
                inception_a=originals_inception,
                inception_b=objective_inception,
                label_a="originals",
                label_b="objective"
            )
        )

        print(
            "\nCreating global visualizations..."
        )

        Visualizer.create_global_visualizations(
            source_features,
            objective_features,
            originals_features,
            Path(args.output)
            / "global"
        )

    conditional_results = None

    if args.originals:

        conditional_results = (
            ImageMetrics.evaluate(
                originals_dir=args.originals,
                objective_dir=args.objective
            )
        )

        print_results(
            conditional_results["summary"],
            "CONDITIONAL METRICS"
        )

        save_conditional_results(
            conditional_results,
            args.output
        )

        Visualizer.create_conditional_visualizations(
            conditional_results,
            args.output
        )

    save_json(
        {
            "source_vs_objective":
            source_vs_objective_metrics,

            "source_vs_originals":
            source_vs_originals_metrics,

            "originals_vs_objective":
            originals_vs_objective_metrics
        },
        Path(args.output)
        / "all_embedding_metrics.json"
    )

    save_run_metadata(
        args,
        device,
        args.output
    )

    ReportGenerator.create_report(
        output_file=(
            Path(args.output)
            / "report.md"
        ),
        source_path=args.source,
        objective_path=args.objective,
        originals_path=args.originals,
        embedding_metrics=source_vs_objective_metrics,
        conditional_metrics=(
            None
            if conditional_results is None
            else conditional_results["summary"]
        )
    )

    print(
        "\nFinished."
    )

    print(
        Path(args.output).resolve()
    )


if __name__ == "__main__":
    main()
