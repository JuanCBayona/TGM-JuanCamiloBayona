import numpy as np

from scipy.linalg import sqrtm

from config import (
    FID_MIN_SAMPLES,
    IS_SPLITS,
    RANDOM_SEED
)


class GenerativeMetrics:
    """
    FID  : Frechet Inception Distance between two datasets.
           Computed on the 2048-d InceptionV3 pool activations
           (no PCA), which is the standard definition.

    IS   : Inception Score of a single dataset.
           Computed on the 1000-d ImageNet softmax outputs.

    Note: this is different from Metrics.frechet_distance, which
    computes a Frechet distance on the PCA-reduced features of the
    configured backbone (resnet/densenet/efficientnet/UNI).
    """

    @staticmethod
    def _check_sample_size(
        features,
        name
    ):

        if len(features) < FID_MIN_SAMPLES:

            print(
                f"WARNING: {name} has only "
                f"{len(features):,} images "
                f"(< {FID_MIN_SAMPLES:,}). "
                f"FID is biased for small sample "
                f"sizes; treat the value as "
                f"relative, not absolute."
            )

    @staticmethod
    def frechet_inception_distance(
        source_features,
        objective_features,
        eps=1e-6
    ):

        GenerativeMetrics._check_sample_size(
            source_features,
            "source"
        )

        GenerativeMetrics._check_sample_size(
            objective_features,
            "objective"
        )

        source_features = np.asarray(
            source_features,
            dtype=np.float64
        )

        objective_features = np.asarray(
            objective_features,
            dtype=np.float64
        )

        mu1 = np.mean(
            source_features,
            axis=0
        )

        mu2 = np.mean(
            objective_features,
            axis=0
        )

        sigma1 = np.cov(
            source_features,
            rowvar=False
        )

        sigma2 = np.cov(
            objective_features,
            rowvar=False
        )

        diff = mu1 - mu2

        covmean, _ = sqrtm(
            sigma1 @ sigma2,
            disp=False
        )

        if not np.isfinite(
            covmean
        ).all():

            print(
                "WARNING: singular product of "
                "covariances, adding epsilon to "
                "the diagonal."
            )

            offset = (
                np.eye(
                    sigma1.shape[0]
                )
                * eps
            )

            covmean = sqrtm(
                (sigma1 + offset)
                @
                (sigma2 + offset)
            )

        if np.iscomplexobj(
            covmean
        ):

            imaginary_part = np.max(
                np.abs(
                    covmean.imag
                )
            )

            if imaginary_part > 1e-3:

                print(
                    f"WARNING: large imaginary "
                    f"component in matrix sqrt "
                    f"({imaginary_part:.3e}), "
                    f"taking the real part."
                )

            covmean = covmean.real

        distance = (
            diff @ diff
            +
            np.trace(sigma1)
            +
            np.trace(sigma2)
            -
            2 * np.trace(covmean)
        )

        return float(
            max(
                0.0,
                distance
            )
        )

    @staticmethod
    def inception_score(
        probabilities,
        splits=IS_SPLITS,
        eps=1e-16
    ):
        """
        Returns (mean, std) of exp(E_x[ KL(p(y|x) || p(y)) ])
        computed over `splits` disjoint, shuffled subsets.
        """

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64
        )

        num_images = len(
            probabilities
        )

        if num_images == 0:

            return (
                float("nan"),
                float("nan")
            )

        splits = max(
            1,
            min(
                splits,
                num_images
            )
        )

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        order = rng.permutation(
            num_images
        )

        probabilities = probabilities[
            order
        ]

        scores = []

        for chunk in np.array_split(
            probabilities,
            splits
        ):

            if len(chunk) == 0:
                continue

            marginal = np.mean(
                chunk,
                axis=0,
                keepdims=True
            )

            kl = chunk * (
                np.log(chunk + eps)
                -
                np.log(marginal + eps)
            )

            kl = np.sum(
                kl,
                axis=1
            )

            scores.append(
                float(
                    np.exp(
                        np.mean(kl)
                    )
                )
            )

        return (
            float(
                np.mean(scores)
            ),
            float(
                np.std(scores)
            )
        )

    @staticmethod
    def evaluate(
        source_features,
        objective_features,
        source_probabilities,
        objective_probabilities,
        source_label="source",
        objective_label="objective"
    ):
        """
        Returns a flat dict of floats so it can be merged straight
        into the embedding metrics dict (json / csv / report).
        """

        results = {}

        print("Computing FID...")

        results["FID"] = (
            GenerativeMetrics
            .frechet_inception_distance(
                source_features,
                objective_features
            )
        )

        print("Computing IS...")

        (
            source_is_mean,
            source_is_std
        ) = GenerativeMetrics.inception_score(
            source_probabilities
        )

        (
            objective_is_mean,
            objective_is_std
        ) = GenerativeMetrics.inception_score(
            objective_probabilities
        )

        results[
            f"IS_{source_label}_mean"
        ] = source_is_mean

        results[
            f"IS_{source_label}_std"
        ] = source_is_std

        results[
            f"IS_{objective_label}_mean"
        ] = objective_is_mean

        results[
            f"IS_{objective_label}_std"
        ] = objective_is_std

        return results
