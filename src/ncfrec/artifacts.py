"""Versioned, validated persistence for trained ncfrec artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from ncfrec.config import ModelConfig
from ncfrec.model import NeuMF

ARTIFACT_VERSION = 2
WEIGHTS_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class ArtifactMetadata:
    """Validated metadata required to reconstruct a model and ID mappings."""

    model_config: ModelConfig
    raw_user_to_index: dict[int, int]
    raw_movie_to_index: dict[int, int]
    movie_titles_by_index: list[str]
    seen_items: dict[int, set[int]]
    weights_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_index_mapping(name: str, mapping: Mapping[int, int], expected_size: int) -> None:
    expected_indices = set(range(expected_size))
    actual_indices = set(mapping.values())
    if len(mapping) != expected_size or actual_indices != expected_indices:
        raise ValueError(f"{name} must map exactly once onto indices 0..{expected_size - 1}")


def save_artifact(
    model: NeuMF,
    artifact_dir: str | Path,
    raw_user_to_index: Mapping[int, int],
    raw_movie_to_index: Mapping[int, int],
    movie_titles_by_index: list[str],
    seen_items: Mapping[int, set[int]],
) -> Path:
    """Publish weights and metadata with a digest that rejects mixed file pairs."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _validate_index_mapping("raw_user_to_index", raw_user_to_index, model.num_users)
    _validate_index_mapping("raw_movie_to_index", raw_movie_to_index, model.num_items)
    if len(movie_titles_by_index) != model.num_items:
        raise ValueError("movie_titles_by_index length must equal model.num_items")

    temporary_weights = artifact_dir / f".{WEIGHTS_FILENAME}.tmp"
    temporary_metadata = artifact_dir / f".{METADATA_FILENAME}.tmp"
    torch.save(model.state_dict(), temporary_weights)
    weights_sha256 = _sha256_file(temporary_weights)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "weights_sha256": weights_sha256,
        "model_config": model.config.to_dict(),
        "raw_user_to_index": {str(key): int(value) for key, value in raw_user_to_index.items()},
        "raw_movie_to_index": {str(key): int(value) for key, value in raw_movie_to_index.items()},
        "movie_titles_by_index": list(movie_titles_by_index),
        "seen_items": {str(key): sorted(int(item) for item in value) for key, value in seen_items.items()},
    }
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # The digest in metadata makes an interrupted two-file publication fail closed.
    os.replace(temporary_weights, artifact_dir / WEIGHTS_FILENAME)
    os.replace(temporary_metadata, artifact_dir / METADATA_FILENAME)
    return artifact_dir


def load_metadata(artifact_dir: str | Path) -> ArtifactMetadata:
    """Read metadata and reject malformed serving indexes at the artifact boundary."""
    metadata_path = Path(artifact_dir) / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing artifact metadata: {metadata_path}")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if raw.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version: {raw.get('artifact_version')}")

    required = {
        "weights_sha256",
        "model_config",
        "raw_user_to_index",
        "raw_movie_to_index",
        "movie_titles_by_index",
        "seen_items",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"artifact metadata is missing fields: {sorted(missing)}")

    config = ModelConfig.from_dict(raw["model_config"])
    users = {int(key): int(value) for key, value in raw["raw_user_to_index"].items()}
    movies = {int(key): int(value) for key, value in raw["raw_movie_to_index"].items()}
    titles = [str(title) for title in raw["movie_titles_by_index"]]
    seen_items = {int(key): {int(item) for item in value} for key, value in raw["seen_items"].items()}
    weights_sha256 = str(raw["weights_sha256"])

    _validate_index_mapping("raw_user_to_index", users, config.num_users)
    _validate_index_mapping("raw_movie_to_index", movies, config.num_items)
    if len(titles) != config.num_items:
        raise ValueError("artifact title count does not match model config")
    if len(weights_sha256) != 64 or any(character not in "0123456789abcdef" for character in weights_sha256):
        raise ValueError("weights_sha256 must be a lowercase SHA-256 digest")

    valid_users = set(range(config.num_users))
    valid_items = set(range(config.num_items))
    if not set(seen_items).issubset(valid_users):
        raise ValueError("seen_items contains an out-of-range user index")
    if any(not items.issubset(valid_items) for items in seen_items.values()):
        raise ValueError("seen_items contains an out-of-range item index")

    return ArtifactMetadata(
        model_config=config,
        raw_user_to_index=users,
        raw_movie_to_index=movies,
        movie_titles_by_index=titles,
        seen_items=seen_items,
        weights_sha256=weights_sha256,
    )


def load_artifact(
    artifact_dir: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[NeuMF, ArtifactMetadata]:
    """Load a trusted local artifact after checking cross-file consistency."""
    artifact_dir = Path(artifact_dir)
    weights_path = artifact_dir / WEIGHTS_FILENAME
    if not weights_path.exists():
        raise FileNotFoundError(f"missing model weights: {weights_path}")
    metadata = load_metadata(artifact_dir)
    if _sha256_file(weights_path) != metadata.weights_sha256:
        raise ValueError("model weights checksum does not match artifact metadata")

    resolved_device = torch.device(device)
    model = NeuMF(metadata.model_config).to(resolved_device)
    try:
        state_dict = torch.load(weights_path, map_location=resolved_device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older Torch versions
        state_dict = torch.load(weights_path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata
