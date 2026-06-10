import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import umap

from io_utils import ensure_dir


class Visualizer:

    @staticmethod
    def pca_plot(
        train_features,
        generated_features,
        output_file
    ):

        combined = np.vstack([
            train_features,
            generated_features
        ])

        pca = PCA(
            n_components=2,
            random_state=42
        )

        embedding = pca.fit_transform(
            combined
        )

        train_emb = embedding[
            :len(train_features)
        ]

        generated_emb = embedding[
            len(train_features):
        ]

        plt.figure(
            figsize=(10, 8)
        )

        plt.scatter(
            train_emb[:, 0],
            train_emb[:, 1],
            s=5,
            alpha=0.5,
            label="Train"
        )

        plt.scatter(
            generated_emb[:, 0],
            generated_emb[:, 1],
            s=5,
            alpha=0.5,
            label="Generated"
        )

        plt.title(
            "PCA Projection"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300
        )

        plt.close()

    @staticmethod
    def umap_plot(
        train_features,
        generated_features,
        output_file
    ):

        combined = np.vstack([
            train_features,
            generated_features
        ])

        reducer = umap.UMAP(
            n_components=2,
            random_state=42
        )

        embedding = reducer.fit_transform(
            combined
        )

        train_emb = embedding[
            :len(train_features)
        ]

        generated_emb = embedding[
            len(train_features):
        ]

        plt.figure(
            figsize=(10, 8)
        )

        plt.scatter(
            train_emb[:, 0],
            train_emb[:, 1],
            s=5,
            alpha=0.5,
            label="Train"
        )

        plt.scatter(
            generated_emb[:, 0],
            generated_emb[:, 1],
            s=5,
            alpha=0.5,
            label="Generated"
        )

        plt.title(
            "UMAP Projection"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300
        )

        plt.close()

    @staticmethod
    def histogram_comparison(
        train_features,
        generated_features,
        output_file
    ):

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            train_features.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            label="Train"
        )

        plt.hist(
            generated_features.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            label="Generated"
        )

        plt.legend()

        plt.title(
            "Feature Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300
        )

        plt.close()

    @staticmethod
    def create_all(
        train_features,
        generated_features,
        output_dir
    ):

        ensure_dir(
            output_dir
        )

        Visualizer.pca_plot(
            train_features,
            generated_features,
            f"{output_dir}/pca.png"
        )

        Visualizer.umap_plot(
            train_features,
            generated_features,
            f"{output_dir}/umap.png"
        )

        Visualizer.histogram_comparison(
            train_features,
            generated_features,
            f"{output_dir}/feature_histogram.png"
        )
