import numpy as np

from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp

from scipy.linalg import sqrtm

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from config import PCA_COMPONENTS


class Metrics:

    @staticmethod
    def reduce_features(
        train_features,
        generated_features
    ):

        combined = np.vstack([
            train_features,
            generated_features
        ])

        n_components = min(
            PCA_COMPONENTS,
            combined.shape[1],
            combined.shape[0] - 1
        )

        pca = PCA(
            n_components=n_components,
            random_state=42
        )

        reduced = pca.fit_transform(
            combined
        )

        train_reduced = reduced[
            :len(train_features)
        ]

        generated_reduced = reduced[
            len(train_features):
        ]

        return (
            train_reduced,
            generated_reduced
        )

    @staticmethod
    def compute_histograms(
        train_features,
        generated_features,
        bins=50
    ):

        train_hists = []
        generated_hists = []

        num_dims = train_features.shape[1]

        for dim in range(num_dims):

            combined = np.concatenate([
                train_features[:, dim],
                generated_features[:, dim]
            ])

            hist_range = (
                combined.min(),
                combined.max()
            )

            h1, _ = np.histogram(
                train_features[:, dim],
                bins=bins,
                range=hist_range,
                density=True
            )

            h2, _ = np.histogram(
                generated_features[:, dim],
                bins=bins,
                range=hist_range,
                density=True
            )

            h1 = h1 + 1e-10
            h2 = h2 + 1e-10

            h1 /= h1.sum()
            h2 /= h2.sum()

            train_hists.append(h1)
            generated_hists.append(h2)

        return (
            train_hists,
            generated_hists
        )

    @staticmethod
    def kl_divergence(
        train_hists,
        generated_hists
    ):

        values = []

        for h1, h2 in zip(
            train_hists,
            generated_hists
        ):

            values.append(
                entropy(h1, h2)
            )

        return float(
            np.mean(values)
        )

    @staticmethod
    def js_divergence(
        train_hists,
        generated_hists
    ):

        values = []

        for h1, h2 in zip(
            train_hists,
            generated_hists
        ):

            m = 0.5 * (
                h1 + h2
            )

            js = (
                0.5 * entropy(h1, m)
                +
                0.5 * entropy(h2, m)
            )

            values.append(js)

        return float(
            np.mean(values)
        )

    @staticmethod
    def emd_distance(
        train_hists,
        generated_hists
    ):

        values = []

        for h1, h2 in zip(
            train_hists,
            generated_hists
        ):

            values.append(
                wasserstein_distance(
                    h1,
                    h2
                )
            )

        return float(
            np.mean(values)
        )

    @staticmethod
    def mmd_distance(
        train_features,
        generated_features,
        gamma=1.0,
        max_samples=2000
    ):

        rng = np.random.default_rng(
            42
        )

        if len(train_features) > max_samples:

            idx = rng.choice(
                len(train_features),
                max_samples,
                replace=False
            )

            train_features = (
                train_features[idx]
            )

        if len(generated_features) > max_samples:

            idx = rng.choice(
                len(generated_features),
                max_samples,
                replace=False
            )

            generated_features = (
                generated_features[idx]
            )

        xx = rbf_kernel(
            train_features,
            train_features,
            gamma=gamma
        )

        yy = rbf_kernel(
            generated_features,
            generated_features,
            gamma=gamma
        )

        xy = rbf_kernel(
            train_features,
            generated_features,
            gamma=gamma
        )

        return float(
            xx.mean()
            +
            yy.mean()
            -
            2 * xy.mean()
        )

    @staticmethod
    def frechet_distance(
        train_features,
        generated_features
    ):

        mu1 = np.mean(
            train_features,
            axis=0
        )

        mu2 = np.mean(
            generated_features,
            axis=0
        )

        sigma1 = np.cov(
            train_features,
            rowvar=False
        )

        sigma2 = np.cov(
            generated_features,
            rowvar=False
        )

        diff = mu1 - mu2

        covmean = sqrtm(
            sigma1 @ sigma2
        )

        if np.iscomplexobj(
            covmean
        ):
            covmean = covmean.real

        distance = (
            diff @ diff
            +
            np.trace(
                sigma1
                +
                sigma2
                -
                2 * covmean
            )
        )

        return float(
            distance
        )

    @staticmethod
    def ks_distance(
        train_features,
        generated_features
    ):

        values = []

        for dim in range(
            train_features.shape[1]
        ):

            ks, _ = ks_2samp(
                train_features[:, dim],
                generated_features[:, dim]
            )

            values.append(
                ks
            )

        return float(
            np.mean(values)
        )

    @staticmethod
    def negative_log_likelihood(
        train_features,
        generated_features,
        n_components=3
    ):

        n_components = min(
            n_components,
            max(
                1,
                len(train_features) // 20
            )
        )

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=1e-4,
            random_state=42
        )

        gmm.fit(
            train_features.astype(
                np.float64
            )
        )

        log_probs = gmm.score_samples(
            generated_features.astype(
                np.float64
            )
        )

        return float(
            -np.mean(log_probs)
        )

    @staticmethod
    def evaluate(
        train_features,
        generated_features
    ):

        (
            train_reduced,
            generated_reduced
        ) = Metrics.reduce_features(
            train_features,
            generated_features
        )

        (
            train_hists,
            generated_hists
        ) = Metrics.compute_histograms(
            train_reduced,
            generated_reduced
        )

        results = {}

        print("Computing KL...")
        results["KL"] = Metrics.kl_divergence(
            train_hists,
            generated_hists
        )

        print("Computing JS...")
        results["JS"] = Metrics.js_divergence(
            train_hists,
            generated_hists
        )

        print("Computing EMD...")
        results["EMD"] = Metrics.emd_distance(
            train_hists,
            generated_hists
        )

        print("Computing MMD...")
        results["MMD"] = Metrics.mmd_distance(
            train_reduced,
            generated_reduced
        )

        print("Computing Frechet...")
        results["Frechet"] = Metrics.frechet_distance(
            train_reduced,
            generated_reduced
        )

        print("Computing KS...")
        results["KS"] = Metrics.ks_distance(
            train_reduced,
            generated_reduced
        )

        print("Computing NLL...")
        results["NLL"] = Metrics.negative_log_likelihood(
            train_reduced,
            generated_reduced
        )

        return results
