"""Artifact and production inference contract tests."""

import json
from pathlib import Path

import pytest
import torch

from ncfrec.artifacts import (
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    load_artifact,
    load_metadata,
    save_artifact,
)
from ncfrec.config import ModelConfig
from ncfrec.model import NeuMF
from ncfrec.recommender import NCFRecommender


def create_artifact(tmp_path: Path) -> tuple[Path, NeuMF]:
    torch.manual_seed(9)
    model = NeuMF(ModelConfig(num_users=2, num_items=4, embedding_dim=4, mlp_layers=(8, 4), dropout=0.0))
    artifact_dir = tmp_path / "artifact"
    save_artifact(
        model,
        artifact_dir,
        raw_user_to_index={10: 0, 20: 1},
        raw_movie_to_index={100: 0, 101: 1, 102: 2, 103: 3},
        movie_titles_by_index=["Zero", "One", "Two", "Three"],
        seen_items={0: {0}, 1: {1}},
    )
    return artifact_dir, model


def read_metadata(artifact_dir: Path) -> tuple[Path, dict]:
    metadata_path = artifact_dir / METADATA_FILENAME
    return metadata_path, json.loads(metadata_path.read_text(encoding="utf-8"))


def test_artifact_round_trip_preserves_predictions(tmp_path: Path) -> None:
    artifact_dir, original = create_artifact(tmp_path)
    restored, metadata = load_artifact(artifact_dir)
    users = torch.tensor([0, 1])
    items = torch.tensor([2, 3])

    assert torch.allclose(original.predict_proba(users, items), restored.predict_proba(users, items))
    assert metadata.movie_titles_by_index[2] == "Two"


def test_recommendations_are_ranked_serializable_and_exclude_seen(tmp_path: Path) -> None:
    artifact_dir, _ = create_artifact(tmp_path)
    service = NCFRecommender.from_artifact(artifact_dir)
    recommendations = service.recommend(user_id=10, k=10)

    assert len(recommendations) == 3
    assert 100 not in {recommendation.movie_id for recommendation in recommendations}
    assert [item.score for item in recommendations] == sorted(
        (item.score for item in recommendations), reverse=True
    )
    assert all(json.dumps(item.to_dict()) for item in recommendations)
    assert service.health() == {"status": "ready", "num_users": 2, "num_items": 4, "device": "cpu"}


def test_service_rejects_unknown_ids_and_invalid_k(tmp_path: Path) -> None:
    artifact_dir, _ = create_artifact(tmp_path)
    service = NCFRecommender.from_artifact(artifact_dir)

    with pytest.raises(KeyError, match="unknown user_id"):
        service.recommend(user_id=999)
    with pytest.raises(KeyError, match="unknown candidate"):
        service.recommend(user_id=10, candidate_movie_ids=[999])
    with pytest.raises(ValueError, match="positive"):
        service.recommend(user_id=10, k=0)


def test_metadata_loader_rejects_unsupported_version(tmp_path: Path) -> None:
    artifact_dir, _ = create_artifact(tmp_path)
    metadata_path, metadata = read_metadata(artifact_dir)
    metadata["artifact_version"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported artifact version"):
        load_metadata(artifact_dir)


def test_metadata_loader_rejects_incomplete_or_duplicate_item_indexes(tmp_path: Path) -> None:
    artifact_dir, _ = create_artifact(tmp_path)
    metadata_path, metadata = read_metadata(artifact_dir)
    metadata["raw_movie_to_index"]["103"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="raw_movie_to_index"):
        load_metadata(artifact_dir)


def test_metadata_loader_rejects_out_of_range_seen_indexes(tmp_path: Path) -> None:
    artifact_dir, _ = create_artifact(tmp_path)
    metadata_path, metadata = read_metadata(artifact_dir)
    metadata["seen_items"]["0"] = [99]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-range item"):
        load_metadata(artifact_dir)


def test_artifact_loader_rejects_weights_metadata_mismatch(tmp_path: Path) -> None:
    artifact_dir, _ = create_artifact(tmp_path)
    weights_path = artifact_dir / WEIGHTS_FILENAME
    weights_path.write_bytes(weights_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        load_artifact(artifact_dir)
