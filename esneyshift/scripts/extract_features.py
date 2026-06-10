import numpy as np
import torch
import timm

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

from config import (
    IMAGE_SIZE,
    UNI_CHECKPOINT
)

from io_utils import get_image_paths


class HistologyDataset(Dataset):

    def __init__(self, folder):

        self.paths = get_image_paths(folder)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.paths[idx]
        ).convert("RGB")

        return self.transform(image)


def load_model(device):

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True
    )

    checkpoint = torch.load(
        UNI_CHECKPOINT,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.eval()

    model.to(device)

    return model


@torch.no_grad()
def extract_embeddings(
    dataset_path,
    batch_size,
    num_workers,
    device
):

    dataset = HistologyDataset(
        dataset_path
    )

    if len(dataset) == 0:

        raise RuntimeError(
            f"No images found in {dataset_path}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )

    model = load_model(device)

    features = []

    print(
        f"\nExtracting features from "
        f"{len(dataset):,} images..."
    )

    for batch in tqdm(loader):

        batch = batch.to(device)

        embeddings = model(batch)

        embeddings = (
            embeddings
            .detach()
            .cpu()
            .numpy()
        )

        features.append(
            embeddings
        )

    features = np.concatenate(
        features,
        axis=0
    )

    print(
        f"Feature matrix shape: "
        f"{features.shape}"
    )

    return features


def save_features(
    features,
    output_file
):

    np.save(
        output_file,
        features
    )


def load_features(
    feature_file
):

    return np.load(
        feature_file
    )
