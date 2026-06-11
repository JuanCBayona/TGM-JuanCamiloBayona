# Dataset Comparison Tool

This tool compares two image datasets using deep feature embeddings extracted from either a pretrained UNI foundation model or a torchvision backbone (ResNet, DenseNet, EfficientNet).

The implementation generates quantitative metrics, visualizations, and a report summarizing the similarity between the source and objective datasets.

---

# Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

---

# Required Inputs

The script requires two datasets:

| Parameter | Description |
|------------|------------|
| `--source` | Reference dataset used to define the target distribution. |
| `--objective` | Dataset to be compared against the source dataset. |

Optionally, a third dataset can be provided:

| Parameter | Description |
|------------|------------|
| `--originals` | Original images corresponding to the generated images in `--objective`. Used to compute image-level conditional metrics. |

---

# Feature Extractor Selection

You must choose **one** feature extractor.

## Option 1: UNI Foundation Model

Provide the UNI checkpoint:

```bash
--checkpoint path/to/pytorch_model.bin
```

## Option 2: Torchvision Backbone

Select one of the supported models:

```bash
resnet18
resnet34
resnet50
resnet101

densenet121
densenet169
densenet201

efficientnet_b0
efficientnet_b1
efficientnet_b2
efficientnet_b3
efficientnet_b4
```

> Do not specify both `--checkpoint` and `--model-type` at the same time.

---

# Parameters

| Parameter | Required | Description |
|------------|------------|------------|
| `--source` | Yes | Source dataset. |
| `--objective` | Yes | Dataset being evaluated. |
| `--checkpoint` | No | UNI checkpoint file. |
| `--model-type` | No | Torchvision model to use. |
| `--originals` | No | Original images for conditional evaluation. |
| `--batch-size` | No | Batch size used during feature extraction. Default: `64`. |
| `--num-workers` | No | Number of dataloader workers. Default: `8`. |
| `--output` | No | Output directory. Default: `results`. |

---

# Example Commands

## Using UNI

```bash
python run.py \
    --checkpoint /mnt/media2/JuanBayona/esneyshift/modelos_fundacionales/UNI/pytorch_model.bin \
    --source /mnt/media2/JuanBayona/DataSets/combined_datasets/train/histopathology \
    --objective /mnt/media2/JuanBayona/DataSets/combined_datasets/val_Reinhard \
    --originals /mnt/media2/JuanBayona/DataSets/combined_datasets/val/histopathology \
    --output normalizacion_UNI
```

## Using ResNet50

```bash
python run.py \
    --model-type resnet50 \
    --source /mnt/media2/JuanBayona/DataSets/combined_datasets/train/histopathology \
    --objective /mnt/media2/JuanBayona/DataSets/combined_datasets/val_Reinhard \
    --originals /mnt/media2/JuanBayona/DataSets/combined_datasets/val/histopathology \
    --output normalizacion_ResNet50
```

---

# Dataset Structure

The source and objective datasets may contain nested folders.

Example:

```text
dataset/
├── class_1/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── class_2/
│   ├── image_003.png
│   └── ...
└── ...
```

Supported image formats:

```text
jpg
jpeg
png
bmp
tif
tiff
webp
```

---

# Conditional Evaluation Requirements

If `--originals` is provided:

- Every image in `--originals` must have a matching image in `--objective`.
- Relative paths must be identical.
- The number of images must match.

Example:

```text
originals/
└── sample/image_001.png

objective/
└── sample/image_001.png
```

---

# Output Files

The output directory will contain:

```text
results/
├── metadata.json
├── metrics.json
├── metrics.csv
├── report.md
├── pca.png
├── umap.png
├── feature_histogram.png
├── cosine_similarity.png
├── conditional_metrics.csv
├── conditional_metrics_summary.json
├── conditional_metrics.png
└── example_pairs.png
```

---

# Generated Results

## metadata.json

Contains information about:

- Feature extractor used
- Dataset paths
- Checkpoint path
- Device used (CPU/GPU)
- Number of images in each dataset

## metrics.json

Embedding-based comparison metrics in JSON format.

## metrics.csv

Same metrics stored in CSV format.

## report.md

Automatically generated summary report.

## pca.png

2D PCA projection of source and objective feature distributions.

## umap.png

2D UMAP projection of source and objective feature distributions.

## feature_histogram.png

Comparison of the feature distributions extracted from both datasets.

## cosine_similarity.png

Visualization of cosine similarity between dataset feature centroids.

## conditional_metrics.csv

Per-image conditional evaluation results.

Generated only when `--originals` is provided.

## conditional_metrics_summary.json

Average conditional metrics across all image pairs.

Generated only when `--originals` is provided.

## conditional_metrics.png

Visualization of the aggregated conditional metrics.

Generated only when `--originals` is provided.

## example_pairs.png

Displays the worst matching image pairs together with a difference map.

Generated only when `--originals` is provided.

---

# Feature Cache

Feature extraction can be expensive.

To avoid recomputing embeddings every time, extracted features are automatically cached in:

```text
cache/
```

If the same dataset and feature extractor are used again, cached embeddings will be loaded automatically.

This significantly reduces runtime for repeated experiments.

---

# Typical Workflow

1. Install the dependencies.
2. Select a feature extractor (UNI or torchvision model).
3. Define the source dataset.
4. Define the objective dataset.
5. Optionally provide the original images for conditional evaluation.
6. Run one of the example commands.
7. Inspect the generated metrics, plots, and report inside the output directory.

For most users, the two example commands shown above are sufficient to run the implementation in less than five minutes.
