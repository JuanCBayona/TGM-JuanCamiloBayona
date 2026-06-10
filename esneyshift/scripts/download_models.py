from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(
    "/mnt/media2/JuanBayona/esneyshift/modelos_fundacionales"
)

MODELS = {
    "UNI": "MahmoodLab/UNI",
    "H0-mini": "bioptimus/H0-mini"
}

for model_name, repo_id in MODELS.items():

    output_dir = BASE_DIR / model_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nDownloading {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir)
    )
