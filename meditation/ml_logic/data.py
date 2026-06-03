import os
import tempfile
from pathlib import Path

from google.cloud import storage
from huggingface_hub import snapshot_download


REPO_ID = "L-FAME-Dataset-Benchmark/L-FAME"
REPO_TYPE = "dataset"
ALLOW_PATTERNS = [
    "derivatives/ml_preproc_data/**",
    "participants.tsv",
]

GCS_BUCKET_NAME = "meditation_bucket_2271"
GCS_DESTINATION_PREFIX = "raw_data"   # dossier cible dans le bucket


def _upload_dir_to_gcs(local_dir: str, bucket_name: str, prefix: str) -> None:
    """Upload récursif d'un répertoire local vers GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_path = Path(local_dir)
    files = list(local_path.rglob("*"))
    files = [f for f in files if f.is_file()]

    print(f"[↑] Upload de {len(files)} fichiers vers gs://{bucket_name}/{prefix}/")

    for file_path in files:
        relative = file_path.relative_to(local_path)
        blob_name = f"{prefix}/{relative}" if prefix else str(relative)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(file_path))
        print(f"    ✓ {blob_name}")

    print(f"[✓] Upload terminé → gs://{bucket_name}/{prefix}/")


def save_data_into_bucket(
    bucket_name: str = GCS_BUCKET_NAME,
    gcs_prefix: str = GCS_DESTINATION_PREFIX,
    repo_id: str = REPO_ID,
    repo_type: str = REPO_TYPE,
    allow_patterns: list[str] = ALLOW_PATTERNS,
) -> str:
    """
    Télécharge les données L-FAME depuis HuggingFace Hub
    et les sauvegarde dans un bucket GCS.

    Flux : HuggingFace → dossier temporaire local → GCS bucket

    Args:
        bucket_name:     Nom du bucket GCS (sans le préfixe gs://).
        gcs_prefix:      Dossier cible dans le bucket.
        repo_id:         Identifiant du repo HuggingFace.
        repo_type:       Type de repo ("dataset", "model", etc.).
        allow_patterns:  Patterns de fichiers à télécharger.

    Returns:
        URI GCS de destination (gs://bucket/prefix).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[↓] Téléchargement depuis '{repo_id}'...")
        print(f"    Patterns  : {allow_patterns}")
        print(f"    Temp local: {tmp_dir}")

        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            local_dir=tmp_dir,
        )

        print(f"[✓] Téléchargement terminé.")

        _upload_dir_to_gcs(tmp_dir, bucket_name, gcs_prefix)

    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}"
    print(f"[✓] Données disponibles dans : {gcs_uri}")
    return gcs_uri


if __name__ == "__main__":
    save_data_into_bucket()
