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
        source_features,
        objective_features
    ):

        combined = np.vstack([
            source_features,
            objective_features
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

        source_reduced = reduced[
            :len(source_features)
        ]

        objective_reduced = reduced[
            len(source_features):
        ]

        return (
            source_reduced,
            objective_reduced
        )

    @staticmethod
    def compute_histograms(
        source_features,
        objective_features,
        bins=50
    ):

        source_hists = []
        objective_hists = []

        num_dims = source_features.shape[1]

        for dim in range(num_dims):

            combined = np.concatenate([
                source_features[:, dim],
                objective_features[:, dim]
            ])

            hist_range = (
                combined.min(),
                combined.max()
            )

            h1, _ = np.histogram(
                source_features[:, dim],
                bins=bins,
                range=hist_range,
                density=True
            )

            h2, _ = np.histogram(
                objective_features[:, dim],
                bins=bins,
                range=hist_range,
                density=True
            )

            h1 += 1e-10
            h2 += 1e-10

            h1 /= h1.sum()
            h2 /= h2.sum()

            source_hists.append(h1)
            objective_hists.append(h2)

        return (
            source_hists,
            objective_hists
        )

    @staticmethod
    def kl_divergence(
        source_hists,
        objective_hists
    ):

        values = []

        for h1, h2 in zip(
            source_hists,
            objective_hists
        ):

            values.append(
                entropy(h1, h2)
            )

        return float(
            np.mean(values)
        )

    @staticmethod
    def js_divergence(
        source_hists,
        objective_hists
    ):

        values = []

        for h1, h2 in zip(
            source_hists,
            objective_hists
        ):

            m = 0.5 * (h1 + h2)

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
        source_hists,
        objective_hists
    ):

        values = []

        for h1, h2 in zip(
            source_hists,
            objective_hists
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
        source_features,
        objective_features,
        gamma=1.0,
        max_samples=2000
    ):

        rng = np.random.default_rng(42)

        if len(source_features) > max_samples:

            idx = rng.choice(
                len(source_features),
                max_samples,
                replace=False
            )

            source_features = source_features[idx]

        if len(objective_features) > max_samples:

            idx = rng.choice(
                len(objective_features),
                max_samples,
                replace=False
            )

            objective_features = objective_features[idx]

        xx = rbf_kernel(
            source_features,
            source_features,
            gamma=gamma
        )

        yy = rbf_kernel(
            objective_features,
            objective_features,
            gamma=gamma
        )

        xy = rbf_kernel(
            source_features,
            objective_features,
            gamma=gamma
        )

        return float(
            xx.mean()
            + yy.mean()
            - 2 * xy.mean()
        )

    @staticmethod
    def frechet_distance(
        source_features,
        objective_features
    ):

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
                + sigma2
                - 2 * covmean
            )
        )

        return float(distance)

    @staticmethod
    def ks_distance(
        source_features,
        objective_features
    ):

        values = []

        for dim in range(
            source_features.shape[1]
        ):

            ks, _ = ks_2samp(
                source_features[:, dim],
                objective_features[:, dim]
            )

            values.append(ks)

        return float(
            np.mean(values)
        )

    @staticmethod
    def negative_log_likelihood(
        source_features,
        objective_features,
        n_components=3
    ):

        n_components = min(
            n_components,
            max(
                1,
                len(source_features) // 20
            )
        )

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=1e-4,
            random_state=42
        )

        gmm.fit(
            source_features.astype(
                np.float64
            )
        )

        log_probs = gmm.score_samples(
            objective_features.astype(
                np.float64
            )
        )

        return float(
            -np.mean(log_probs)
        )

    @staticmethod
    def cosine_similarity(
        source_features,
        objective_features
    ):

        source_mean = np.mean(
            source_features,
            axis=0
        )

        objective_mean = np.mean(
            objective_features,
            axis=0
        )

        numerator = np.dot(
            source_mean,
            objective_mean
        )

        denominator = (
            np.linalg.norm(
                source_mean
            )
            *
            np.linalg.norm(
                objective_mean
            )
        )

        return float(
            numerator /
            (denominator + 1e-12)
        )

    @staticmethod
    def evaluate(
        source_features,
        objective_features
    ):

        (
            source_reduced,
            objective_reduced
        ) = Metrics.reduce_features(
            source_features,
            objective_features
        )

        (
            source_hists,
            objective_hists
        ) = Metrics.compute_histograms(
            source_reduced,
            objective_reduced
        )

        results = {}

        print("Computing KL...")
        results["KL"] = Metrics.kl_divergence(
            source_hists,
            objective_hists
        )

        print("Computing JS...")
        results["JS"] = Metrics.js_divergence(
            source_hists,
            objective_hists
        )

        print("Computing EMD...")
        results["EMD"] = Metrics.emd_distance(
            source_hists,
            objective_hists
        )

        print("Computing MMD...")
        results["MMD"] = Metrics.mmd_distance(
            source_reduced,
            objective_reduced
        )

        print("Computing Frechet...")
        results["Frechet"] = Metrics.frechet_distance(
            source_reduced,
            objective_reduced
        )

        print("Computing KS...")
        results["KS"] = Metrics.ks_distance(
            source_reduced,
            objective_reduced
        )

        print("Computing NLL...")
        results["NLL"] = Metrics.negative_log_likelihood(
            source_reduced,
            objective_reduced
        )

        print("Computing Cosine Similarity...")
        results["CosineSimilarity"] = (
            Metrics.cosine_similarity(
                source_features,
                objective_features
            )
        )

        return results
