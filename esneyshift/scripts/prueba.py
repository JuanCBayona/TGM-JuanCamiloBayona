import torch

from extract_features import extract_embeddings
from metrics import Metrics

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

train_features = extract_embeddings(
    dataset_path="/mnt/media2/JuanBayona/Fractal/combined_dataset/val/histopathology",
    batch_size=64,
    num_workers=8,
    device=device
)

generated_features = extract_embeddings(
    dataset_path="/mnt/media2/JuanBayona/Fractal/combined_dataset/val/histopathology",
    batch_size=64,
    num_workers=8,
    device=device
)

results = Metrics.evaluate(
    train_features,
    generated_features
)

print(results)
