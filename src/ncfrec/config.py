"""Typed configuration for data preparation, NeuMF, and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Controls implicit-feedback conversion and negative sampling."""

    positive_rating: float = 4.0
    negatives_per_positive: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.5 <= self.positive_rating <= 5.0:
            raise ValueError("positive_rating must be between 0.5 and 5.0")
        if self.negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be at least 1")


@dataclass(frozen=True)
class ModelConfig:
    """NeuMF architecture parameters."""

    num_users: int
    num_items: int
    embedding_dim: int = 32
    mlp_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.num_users < 1 or self.num_items < 1:
            raise ValueError("num_users and num_items must be positive")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if not self.mlp_layers or any(size < 1 for size in self.mlp_layers):
            raise ValueError("mlp_layers must contain positive layer sizes")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        normalized = dict(values)
        normalized["mlp_layers"] = tuple(normalized["mlp_layers"])
        return cls(**normalized)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization parameters used by the training notebook."""

    epochs: int = 5
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    evaluation_k: int = 10

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if self.evaluation_k < 1:
            raise ValueError("evaluation_k must be positive")
