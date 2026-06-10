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

from visualization import Visualizer

from io_utils import ensure_dir

from config import CACHE_DIR


def get_feature_file(
    dataset_path
):

    path_hash = hashlib.md5(
        str(dataset_path).encode()
    ).hexdigest()[:12]

    return (
        CACHE_DIR /
        f"{path_hash}.npy"
    )


def load_or_extract(
    dataset_path,
    batch_size,
    num_workers,
    device
):

    feature_file = get_feature_file(
        dataset_path
    )

    if feature_file.exists():

        print(
            f"\nLoading cached features:"
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
        f"\nExtracting features from:"
    )

    print(
        dataset_path
    )

    features = extract_embeddings(
        dataset_path,
        batch_size,
        num_workers,
        device
    )

    save_features(
        features,
        feature_file
    )

    print(
        f"\nSaved cache:"
    )

    print(
        feature_file
    )

    return features


def save_results(
    results,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    with open(
        output_dir / "metrics.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    pd.DataFrame(
        [results]
    ).to_csv(
        output_dir / "metrics.csv",
        index=False
    )


def print_results(
    results
):

    print("\n")
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)

    for key, value in results.items():

        print(
            f"{key:<15} : "
            f"{value:.6f}"
        )

    print("=" * 50)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare training and generated "
            "datasets using UNI embeddings"
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Training dataset path"
    )

    parser.add_argument(
        "--generated",
        required=True,
        help="Generated dataset path"
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
        f"\nUsing device: {device}"
    )

    train_features = load_or_extract(
        args.train,
        args.batch_size,
        args.num_workers,
        device
    )

    generated_features = load_or_extract(
        args.generated,
        args.batch_size,
        args.num_workers,
        device
    )

    print(
        "\nTrain feature shape:"
    )

    print(
        train_features.shape
    )

    print(
        "\nGenerated feature shape:"
    )

    print(
        generated_features.shape
    )

    results = Metrics.evaluate(
        train_features,
        generated_features
    )

    print_results(
        results
    )

    save_results(
        results,
        args.output
    )

    print(
        "\nCreating visualizations..."
    )

    Visualizer.create_all(
        train_features,
        generated_features,
        args.output
    )

    print(
        "\nFinished."
    )

    print(
        "\nResults saved in:"
    )

    print(
        Path(args.output).resolve()
    )


if __name__ == "__main__":
    main()
