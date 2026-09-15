"""NeuMF architecture contract tests."""

import pytest
import torch

from ncfrec.config import ModelConfig
from ncfrec.model import NeuMF


def make_model() -> NeuMF:
    return NeuMF(ModelConfig(num_users=3, num_items=5, embedding_dim=4, mlp_layers=(8, 4), dropout=0.0))


def test_forward_returns_one_finite_logit_per_pair() -> None:
    model = make_model()
    logits = model(torch.tensor([0, 1, 2]), torch.tensor([4, 3, 2]))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_predict_proba_is_bounded_and_does_not_track_gradients() -> None:
    model = make_model()
    probabilities = model.predict_proba(torch.tensor([0, 1]), torch.tensor([1, 2]))

    assert torch.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert not probabilities.requires_grad


def test_forward_rejects_mismatched_pair_shapes() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="same shape"):
        model(torch.tensor([0, 1]), torch.tensor([1]))
