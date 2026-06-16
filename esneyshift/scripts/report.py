from pathlib import Path

import json


class ReportGenerator:

    @staticmethod
    def _load_metrics_if_exists(
        metrics_file
    ):

        metrics_file = Path(
            metrics_file
        )

        if not metrics_file.exists():

            return None

        with open(
            metrics_file,
            "r"
        ) as f:

            return json.load(
                f
            )

    @staticmethod
    def _write_metrics_section(
        lines,
        title,
        metrics
    ):

        if metrics is None:
            return

        lines.append(
            f"## {title}\n"
        )

        for key, value in metrics.items():

            if value is None:
                continue

            lines.append(
                f"- {key}: "
                f"{value:.6f}"
            )

        lines.append("")

    @staticmethod
    def _write_interpretation(
        lines,
        source_vs_objective,
        source_vs_originals
    ):

        if (
            source_vs_objective is None
            or
            source_vs_originals is None
        ):
            return

        lines.append(
            "## Distribution Shift Analysis\n"
        )

        lower_is_better = [
            "KL",
            "JS",
            "EMD",
            "MMD",
            "Frechet",
            "KS",
            "NLL"
        ]

        improvements = []

        for metric in lower_is_better:

            if (
                metric not in source_vs_originals
                or
                metric not in source_vs_objective
            ):
                continue

            before = (
                source_vs_originals[
                    metric
                ]
            )

            after = (
                source_vs_objective[
                    metric
                ]
            )

            if abs(before) < 1e-12:
                continue

            improvement = (
                (
                    before
                    -
                    after
                )
                /
                abs(before)
            ) * 100.0

            improvements.append(
                improvement
            )

            lines.append(
                f"- {metric}: "
                f"{improvement:.2f}% "
                f"improvement"
            )

        if (
            "CosineSimilarity"
            in source_vs_originals
            and
            "CosineSimilarity"
            in source_vs_objective
        ):

            before = (
                source_vs_originals[
                    "CosineSimilarity"
                ]
            )

            after = (
                source_vs_objective[
                    "CosineSimilarity"
                ]
            )

            if abs(before) > 1e-12:

                improvement = (
                    (
                        after
                        -
                        before
                    )
                    /
                    abs(before)
                ) * 100.0

                improvements.append(
                    improvement
                )

                lines.append(
                    "- CosineSimilarity: "
                    f"{improvement:.2f}% "
                    "improvement"
                )

        lines.append("")

        if len(improvements):

            normalization_score = (
                sum(improvements)
                /
                len(improvements)
            )

            normalization_score = max(
                0.0,
                min(
                    100.0,
                    normalization_score
                )
            )

            lines.append(
                "## Normalization Score\n"
            )

            lines.append(
                f"Overall Score: "
                f"{normalization_score:.2f}"
                f"/100"
            )

            lines.append("")

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

        output_dir = (
            output_file.parent
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

        source_vs_objective = (
            ReportGenerator
            ._load_metrics_if_exists(
                output_dir
                /
                "source_vs_objective"
                /
                "metrics.json"
            )
        )

        source_vs_originals = (
            ReportGenerator
            ._load_metrics_if_exists(
                output_dir
                /
                "source_vs_originals"
                /
                "metrics.json"
            )
        )

        originals_vs_objective = (
            ReportGenerator
            ._load_metrics_if_exists(
                output_dir
                /
                "originals_vs_objective"
                /
                "metrics.json"
            )
        )

        if source_vs_objective is None:

            source_vs_objective = (
                embedding_metrics
            )

        ReportGenerator._write_metrics_section(
            lines,
            "Source vs Objective",
            source_vs_objective
        )

        ReportGenerator._write_metrics_section(
            lines,
            "Source vs Originals",
            source_vs_originals
        )

        ReportGenerator._write_metrics_section(
            lines,
            "Originals vs Objective",
            originals_vs_objective
        )

        ReportGenerator._write_interpretation(
            lines,
            source_vs_objective,
            source_vs_originals
        )

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

        lines.append("")

        lines.append(
            "### source_vs_objective"
        )

        lines.extend([
            "- pca.png",
            "- umap.png",
            "- feature_histogram.png",
            "- cosine_similarity.png",
            "- metrics.json",
            "- metrics.csv"
        ])

        lines.append("")

        if source_vs_originals is not None:

            lines.append(
                "### source_vs_originals"
            )

            lines.extend([
                "- pca.png",
                "- umap.png",
                "- feature_histogram.png",
                "- cosine_similarity.png",
                "- metrics.json",
                "- metrics.csv"
            ])

            lines.append("")

        if originals_vs_objective is not None:

            lines.append(
                "### originals_vs_objective"
            )

            lines.extend([
                "- pca.png",
                "- umap.png",
                "- feature_histogram.png",
                "- cosine_similarity.png",
                "- metrics.json",
                "- metrics.csv"
            ])

            lines.append("")

            lines.append(
                "### global"
            )

            lines.extend([
                "- global_pca.png",
                "- global_umap.png"
            ])

            lines.append("")

        if conditional_metrics:

            lines.append(
                "### conditional"
            )

            lines.extend([
                "- conditional_metrics.csv",
                "- conditional_metrics_summary.json",
                "- conditional_metrics.png",
                "- example_pairs.png"
            ])

            lines.append("")

        with open(
            output_file,
            "w"
        ) as f:

            f.write(
                "\n".join(lines)
            )
