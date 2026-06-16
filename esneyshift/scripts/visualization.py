from pathlib import Path
from io_utils import ensure_dir
from config import EXAMPLE_PAIRS
from image_metrics import ImageMetrics

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from PIL import Image

MAX_POINTS_PER_DATASET = 5000

class Visualizer:

    @staticmethod
    def _subsample(features, max_points=MAX_POINTS_PER_DATASET):
        if len(features) <= max_points:
            return features
        rng = np.random.default_rng(42)
        indices = rng.choice(len(features), size=max_points, replace=False)
        return features[indices]

    @staticmethod
    def pca_plot(source_features, objective_features, output_file):
        combined = np.vstack([source_features, objective_features])
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(combined)

        source_emb = embedding[:len(source_features)]
        objective_emb = embedding[len(source_features):]

        plt.figure(figsize=(10, 8))
        plt.scatter(source_emb[:, 0], source_emb[:, 1], s=5, alpha=0.5, label="Source")
        plt.scatter(objective_emb[:, 0], objective_emb[:, 1], s=5, alpha=0.5, label="Objective")
        plt.legend()
        plt.title("PCA Projection")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def umap_plot(source_features, objective_features, output_file):
        combined = np.vstack([source_features, objective_features])
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(combined)

        source_emb = embedding[:len(source_features)]
        objective_emb = embedding[len(source_features):]

        plt.figure(figsize=(10, 8))
        plt.scatter(source_emb[:, 0], source_emb[:, 1], s=5, alpha=0.5, label="Source")
        plt.scatter(objective_emb[:, 0], objective_emb[:, 1], s=5, alpha=0.5, label="Objective")
        plt.legend()
        plt.title("UMAP Projection")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def global_pca_plot(source_features, objective_features, originals_features, output_file):
        source_features = Visualizer._subsample(source_features)
        objective_features = Visualizer._subsample(objective_features)
        originals_features = Visualizer._subsample(originals_features)

        combined = np.vstack([source_features, objective_features, originals_features])
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(combined)

        n_source = len(source_features)
        n_objective = len(objective_features)

        source_emb = embedding[:n_source]
        objective_emb = embedding[n_source : n_source + n_objective]
        originals_emb = embedding[n_source + n_objective:]

        plt.figure(figsize=(10, 8))
        plt.scatter(source_emb[:, 0], source_emb[:, 1], s=5, alpha=0.5, label="Source")
        plt.scatter(originals_emb[:, 0], originals_emb[:, 1], s=5, alpha=0.5, label="Originals")
        plt.scatter(objective_emb[:, 0], objective_emb[:, 1], s=5, alpha=0.5, label="Objective")
        plt.legend()
        plt.title("Global PCA Projection")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def global_umap_plot(source_features, objective_features, originals_features, output_file):
        source_features = Visualizer._subsample(source_features)
        objective_features = Visualizer._subsample(objective_features)
        originals_features = Visualizer._subsample(originals_features)

        combined = np.vstack([source_features, objective_features, originals_features])
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(combined)

        n_source = len(source_features)
        n_objective = len(objective_features)

        source_emb = embedding[:n_source]
        objective_emb = embedding[n_source : n_source + n_objective]
        originals_emb = embedding[n_source + n_objective:]

        plt.figure(figsize=(10, 8))
        plt.scatter(source_emb[:, 0], source_emb[:, 1], s=5, alpha=0.5, label="Source")
        plt.scatter(originals_emb[:, 0], originals_emb[:, 1], s=5, alpha=0.5, label="Originals")
        plt.scatter(objective_emb[:, 0], objective_emb[:, 1], s=5, alpha=0.5, label="Objective")
        plt.legend()
        plt.title("Global UMAP Projection")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def histogram_comparison(source_features, objective_features, output_file):
        plt.figure(figsize=(10, 6))
        plt.hist(source_features.flatten(), bins=100, density=True, alpha=0.5, label="Source")
        plt.hist(objective_features.flatten(), bins=100, density=True, alpha=0.5, label="Objective")
        plt.legend()
        plt.title("Feature Distribution")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def cosine_similarity_plot(cosine_value, output_file):
        plt.figure(figsize=(8, 4))
        plt.bar(["Cosine Similarity"], [cosine_value])
        plt.axhline(1.0, linestyle="--", label="Identical")
        plt.axhline(0.9, linestyle="--", label="Similar")
        plt.axhline(0.5, linestyle="--", label="Different")
        plt.axhline(0.0, linestyle="--", label="Orthogonal")
        plt.ylim(0, 1.05)
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def conditional_metrics_plot(summary, output_file):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].bar(["PSNR"], [summary["PSNR_mean"]])
        axes[0].set_title("PSNR")
        
        axes[1].bar(["SSIM"], [summary["SSIM_mean"]])
        axes[1].set_title("SSIM")
        
        axes[2].bar(["MSE"], [summary["MSE_mean"]])
        axes[2].set_title("MSE")
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def example_pairs_plot(worst_pairs, output_file):
        n = min(EXAMPLE_PAIRS, len(worst_pairs))
        if n == 0:
            return

        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, pair in enumerate(worst_pairs[:n]):
            original = np.array(Image.open(pair["original_path"]).convert("RGB"))
            generated = np.array(Image.open(pair["generated_path"]).convert("RGB"))

            if original.shape != generated.shape:
                generated = np.array(
                    Image.fromarray(generated).resize((original.shape[1], original.shape[0]))
                )

            diff = np.abs(original.astype(np.float32) - generated.astype(np.float32))
            diff = diff / (diff.max() + 1e-8)

            axes[row][0].imshow(original)
            axes[row][0].set_title("Original")

            axes[row][1].imshow(generated)
            axes[row][1].set_title(
                f"Generated\nSSIM={pair['ssim']:.3f}\nPSNR={pair['psnr']:.2f}\nMSE={pair['mse']:.2f}"
            )

            axes[row][2].imshow(diff)
            axes[row][2].set_title("Difference Map")

            for col in range(3):
                axes[row][col].axis("off")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def create_embedding_visualizations(source_features, objective_features, metrics, output_dir):
        ensure_dir(output_dir)
        Visualizer.pca_plot(source_features, objective_features, f"{output_dir}/pca.png")
        Visualizer.umap_plot(source_features, objective_features, f"{output_dir}/umap.png")
        Visualizer.histogram_comparison(source_features, objective_features, f"{output_dir}/feature_histogram.png")
        Visualizer.cosine_similarity_plot(metrics["CosineSimilarity"], f"{output_dir}/cosine_similarity.png")

    @staticmethod
    def create_global_visualizations(source_features, objective_features, originals_features, output_dir):
        ensure_dir(output_dir)
        Visualizer.global_pca_plot(
            source_features, objective_features, originals_features, Path(output_dir) / "global_pca.png"
        )
        Visualizer.global_umap_plot(
            source_features, objective_features, originals_features, Path(output_dir) / "global_umap.png"
        )

    @staticmethod
    def create_conditional_visualizations(conditional_results, output_dir):
        ensure_dir(output_dir)
        Visualizer.conditional_metrics_plot(
            conditional_results["summary"], f"{output_dir}/conditional_metrics.png"
        )
        worst_pairs = ImageMetrics.get_worst_pairs(conditional_results["pair_results"], EXAMPLE_PAIRS)
        Visualizer.example_pairs_plot(worst_pairs, f"{output_dir}/example_pairs.png")
