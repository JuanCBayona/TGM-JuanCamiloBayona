import numpy as np
import torch
import timm

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,

    densenet121,
    densenet169,
    densenet201,

    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,

    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,

    DenseNet121_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,

    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights
)

from config import IMAGE_SIZE

from io_utils import get_image_paths


class HistologyDataset(Dataset):

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
                    IMAGE_SIZE,
                    IMAGE_SIZE
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


def load_uni_model(
    checkpoint_path,
    device
):

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.eval()

    model.to(
        device
    )

    return model


def load_torchvision_model(
    model_type,
    device
):

    if model_type == "resnet18":

        model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        model.fc = torch.nn.Identity()

    elif model_type == "resnet34":

        model = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1
        )

        model.fc = torch.nn.Identity()

    elif model_type == "resnet50":

        model = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        model.fc = torch.nn.Identity()

    elif model_type == "resnet101":

        model = resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2
        )

        model.fc = torch.nn.Identity()

    elif model_type == "densenet121":

        model = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        model.classifier = torch.nn.Identity()

    elif model_type == "densenet169":

        model = densenet169(
            weights=DenseNet169_Weights.IMAGENET1K_V1
        )

        model.classifier = torch.nn.Identity()

    elif model_type == "densenet201":

        model = densenet201(
            weights=DenseNet201_Weights.IMAGENET1K_V1
        )

        model.classifier = torch.nn.Identity()

    elif model_type == "efficientnet_b0":

        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        model.classifier = torch.nn.Identity()

    elif model_type == "efficientnet_b1":

        model = efficientnet_b1(
            weights=EfficientNet_B1_Weights.IMAGENET1K_V2
        )

        model.classifier = torch.nn.Identity()

    elif model_type == "efficientnet_b2":

        model = efficientnet_b2(
            weights=EfficientNet_B2_Weights.IMAGENET1K_V1
        )

        model.classifier = torch.nn.Identity()

    elif model_type == "efficientnet_b3":

        model = efficientnet_b3(
            weights=EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        model.classifier = torch.nn.Identity()

    elif model_type == "efficientnet_b4":

        model = efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        model.classifier = torch.nn.Identity()

    else:

        raise ValueError(
            f"Unsupported model type: "
            f"{model_type}"
        )

    model.eval()

    model.to(
        device
    )

    return model


def load_model(
    checkpoint_path,
    model_type,
    device
):

    if (
        checkpoint_path is not None
        and model_type is not None
    ):

        raise ValueError(
            "Specify either "
            "--checkpoint "
            "or "
            "--model-type, "
            "not both."
        )

    if (
        checkpoint_path is None
        and model_type is None
    ):

        raise ValueError(
            "You must specify "
            "either "
            "--checkpoint "
            "or "
            "--model-type."
        )

    if checkpoint_path is not None:

        print(
            "\nUsing UNI model"
        )

        return load_uni_model(
            checkpoint_path,
            device
        )

    print(
        f"\nUsing torchvision model: "
        f"{model_type}"
    )

    return load_torchvision_model(
        model_type,
        device
    )


@torch.no_grad()
def extract_embeddings(
    dataset_path,
    checkpoint_path,
    model_type,
    batch_size,
    num_workers,
    device
):

    dataset = HistologyDataset(
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

    model = load_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device
    )

    features = []

    print(
        f"\nExtracting features from "
        f"{len(dataset):,} images..."
    )

    print(
        f"Total batches: "
        f"{len(loader):,}"
    )

    for batch in tqdm(
        loader,
        desc="Extracting embeddings",
        unit="batch"
    ):

        batch = batch.to(
            device
        )

        embeddings = model(
            batch
        )

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
