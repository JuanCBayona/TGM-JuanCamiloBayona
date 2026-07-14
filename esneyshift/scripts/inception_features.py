import hashlib

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

from torchvision.models import (
    inception_v3,
    Inception_V3_Weights
)

from config import (
    CACHE_DIR,
    INCEPTION_IMAGE_SIZE,
    INCEPTION_FEATURE_DIM,
    INCEPTION_NUM_CLASSES
)

from io_utils import get_image_paths


class InceptionDataset(Dataset):

    def __init__(
        self,
        folder
    ):

        self.paths = get_image_paths(
            folder
        )

        self.transform = transforms.Compose([
            transforms.Resize(
                (
                    INCEPTION_IMAGE_SIZE,
                    INCEPTION_IMAGE_SIZE
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(
                    0.485,
                    0.456,
                    0.406
                ),
                std=(
                    0.229,
                    0.224,
                    0.225
                )
            )
        ])

    def __len__(self):

        return len(
            self.paths
        )

    def __getitem__(
        self,
        idx
    ):

        image = Image.open(
            self.paths[idx]
        ).convert(
            "RGB"
        )

        return self.transform(
            image
        )


def load_inception_model(
    device
):

    model = inception_v3(
        weights=(
            Inception_V3_Weights
            .IMAGENET1K_V1
        )
    )

    model.eval()

    model.to(
        device
    )

    return model


@torch.no_grad()
def extract_inception_activations(
    dataset_path,
    batch_size,
    num_workers,
    device
):
    """
    Runs InceptionV3 (ImageNet weights) over a folder of images
    and returns, in a single pass:

        pool_features : (N, 2048) global average pooled activations
                        -> used for FID
        probabilities : (N, 1000) softmax over ImageNet classes
                        -> used for the Inception Score
    """

    dataset = InceptionDataset(
        dataset_path
    )

    if len(dataset) == 0:

        raise RuntimeError(
            f"No images found in "
            f"{dataset_path}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device == "cuda"
        )
    )

    model = load_inception_model(
        device
    )

    pooled_output = {}

    def pool_hook(
        module,
        module_input,
        module_output
    ):

        pooled_output["value"] = module_output

    hook_handle = (
        model
        .avgpool
        .register_forward_hook(
            pool_hook
        )
    )

    pool_features = []
    probabilities = []

    print(
        f"\nExtracting InceptionV3 activations from "
        f"{len(dataset):,} images..."
    )

    print(
        f"Total batches: "
        f"{len(loader):,}"
    )

    try:

        for batch in tqdm(
            loader,
            desc="Extracting inception activations",
            unit="batch"
        ):

            batch = batch.to(
                device
            )

            logits = model(
                batch
            )

            features = torch.flatten(
                pooled_output["value"],
                1
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            pool_features.append(
                features
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            probabilities.append(
                probs
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

    finally:

        hook_handle.remove()

    pool_features = np.concatenate(
        pool_features,
        axis=0
    )

    probabilities = np.concatenate(
        probabilities,
        axis=0
    )

    print(
        f"Inception feature matrix shape: "
        f"{pool_features.shape}"
    )

    print(
        f"Inception probability matrix shape: "
        f"{probabilities.shape}"
    )

    return (
        pool_features,
        probabilities
    )


def get_inception_cache_file(
    dataset_path
):

    cache_key = (
        str(dataset_path)
        +
        "inception_v3"
        +
        str(INCEPTION_IMAGE_SIZE)
        +
        str(INCEPTION_FEATURE_DIM)
        +
        str(INCEPTION_NUM_CLASSES)
    )

    path_hash = hashlib.md5(
        cache_key.encode()
    ).hexdigest()[:12]

    return (
        CACHE_DIR
        /
        f"{path_hash}_inception.npz"
    )


def save_inception_activations(
    pool_features,
    probabilities,
    output_file
):

    np.savez_compressed(
        output_file,
        pool_features=pool_features,
        probabilities=probabilities
    )


def load_inception_activations(
    cache_file
):

    data = np.load(
        cache_file
    )

    return (
        data["pool_features"],
        data["probabilities"]
    )


def load_or_extract_inception(
    dataset_path,
    batch_size,
    num_workers,
    device
):

    cache_file = get_inception_cache_file(
        dataset_path
    )

    if cache_file.exists():

        print(
            "\nLoading cached inception activations:"
        )

        print(
            cache_file
        )

        (
            pool_features,
            probabilities
        ) = load_inception_activations(
            cache_file
        )

        print(
            f"Loaded shapes: "
            f"{pool_features.shape} "
            f"{probabilities.shape}"
        )

        return (
            pool_features,
            probabilities
        )

    print(
        "\nExtracting inception activations from:"
    )

    print(
        dataset_path
    )

    (
        pool_features,
        probabilities
    ) = extract_inception_activations(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )

    save_inception_activations(
        pool_features,
        probabilities,
        cache_file
    )

    print(
        "\nSaved inception cache:"
    )

    print(
        cache_file
    )

    return (
        pool_features,
        probabilities
    )
