"""ncfrec: neural collaborative filtering for movie recommendations."""

from ncfrec.config import DataConfig, ModelConfig, TrainingConfig
from ncfrec.model import NeuMF
from ncfrec.recommender import NCFRecommender, Recommendation

__all__ = [
    "DataConfig",
    "ModelConfig",
    "NCFRecommender",
    "NeuMF",
    "Recommendation",
    "TrainingConfig",
]

__version__ = "0.1.0"
