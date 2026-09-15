"""Load-once inference service for personalized Top-N recommendations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ncfrec.artifacts import ArtifactMetadata, load_artifact
from ncfrec.model import NeuMF


@dataclass(frozen=True)
class Recommendation:
    """A ranked movie recommendation returned to an application layer."""

    movie_id: int
    title: str
    score: float

    def to_dict(self) -> dict[str, int | str | float]:
        return asdict(self)


class NCFRecommender:
    """Stateful inference facade that loads model weights only once."""

    def __init__(self, model: NeuMF, metadata: ArtifactMetadata, device: torch.device) -> None:
        self.model = model
        self.metadata = metadata
        self.device = device
        self.index_to_raw_movie = [0] * metadata.model_config.num_items
        for raw_movie_id, item_idx in metadata.raw_movie_to_index.items():
            self.index_to_raw_movie[item_idx] = raw_movie_id

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> NCFRecommender:
        resolved_device = torch.device(device)
        model, metadata = load_artifact(artifact_dir, resolved_device)
        return cls(model=model, metadata=metadata, device=resolved_device)

    @torch.inference_mode()
    def predict_interaction(self, user_id: int, movie_id: int) -> float:
        """Score one known external user/movie pair."""
        user_idx = self._user_index(user_id)
        try:
            item_idx = self.metadata.raw_movie_to_index[int(movie_id)]
        except KeyError as error:
            raise KeyError(f"unknown movie_id: {movie_id}") from error
        users = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        items = torch.tensor([item_idx], dtype=torch.long, device=self.device)
        return float(self.model.predict_proba(users, items).item())

    @torch.inference_mode()
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        candidate_movie_ids: Iterable[int] | None = None,
        exclude_seen: bool = True,
    ) -> list[Recommendation]:
        """Return highest-scoring movies for a known user."""
        if k < 1:
            raise ValueError("k must be positive")
        user_idx = self._user_index(user_id)

        if candidate_movie_ids is None:
            candidate_indices = list(range(self.metadata.model_config.num_items))
        else:
            raw_candidates = list(dict.fromkeys(int(movie_id) for movie_id in candidate_movie_ids))
            unknown = [movie_id for movie_id in raw_candidates if movie_id not in self.metadata.raw_movie_to_index]
            if unknown:
                raise KeyError(f"unknown candidate movie_id values: {unknown[:5]}")
            candidate_indices = [self.metadata.raw_movie_to_index[movie_id] for movie_id in raw_candidates]

        if exclude_seen:
            seen = self.metadata.seen_items.get(user_idx, set())
            candidate_indices = [item_idx for item_idx in candidate_indices if item_idx not in seen]
        if not candidate_indices:
            return []

        items = torch.tensor(candidate_indices, dtype=torch.long, device=self.device)
        users = torch.full_like(items, user_idx)
        scores = self.model.predict_proba(users, items)
        result_count = min(k, len(candidate_indices))
        top_scores, top_positions = torch.topk(scores, k=result_count)

        recommendations: list[Recommendation] = []
        for score, position in zip(top_scores.tolist(), top_positions.tolist(), strict=True):
            item_idx = candidate_indices[position]
            recommendations.append(
                Recommendation(
                    movie_id=self.index_to_raw_movie[item_idx],
                    title=self.metadata.movie_titles_by_index[item_idx],
                    score=float(score),
                )
            )
        return recommendations

    def health(self) -> dict[str, int | str]:
        """Expose lightweight readiness metadata for an application health check."""
        return {
            "status": "ready",
            "num_users": self.metadata.model_config.num_users,
            "num_items": self.metadata.model_config.num_items,
            "device": str(self.device),
        }

    def _user_index(self, user_id: int) -> int:
        try:
            return self.metadata.raw_user_to_index[int(user_id)]
        except KeyError as error:
            raise KeyError(f"unknown user_id: {user_id}") from error
