from pathlib import Path


class ReportGenerator:

    @staticmethod
    def create_report(
        output_file,
        source_path,
        objective_path,
        originals_path,
        embedding_metrics,
        conditional_metrics=None
    ):

        output_file = Path(
            output_file
        )

        lines = []

        lines.append(
            "# Dataset Comparison Report\n"
        )

        lines.append(
            "## Datasets\n"
        )

        lines.append(
            f"- Source: {source_path}"
        )

        lines.append(
            f"- Objective: {objective_path}"
        )

        lines.append(
            f"- Originals: "
            f"{originals_path}"
        )

        lines.append("")

        lines.append(
            "## Embedding Metrics\n"
        )

        for key, value in (
            embedding_metrics.items()
        ):

            lines.append(
                f"- {key}: "
                f"{value:.6f}"
            )

        lines.append("")

        if conditional_metrics:

            lines.append(
                "## Conditional Metrics\n"
            )

            for key, value in (
                conditional_metrics.items()
            ):

                lines.append(
                    f"- {key}: "
                    f"{value:.6f}"
                )

            lines.append("")

        lines.append(
            "## Generated Files\n"
        )

        lines.extend([
            "- pca.png",
            "- umap.png",
            "- feature_histogram.png",
            "- cosine_similarity.png"
        ])

        if conditional_metrics:

            lines.extend([
                "- conditional_metrics.png",
                "- example_pairs.png"
            ])

        with open(
            output_file,
            "w"
        ) as f:

            f.write(
                "\n".join(lines)
            )
