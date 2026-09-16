"""Configuration precedence and validation tests."""

import pytest

from ncfrec.config import TrainingConfig


def test_training_config_from_env_uses_class_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCFREC_EPOCHS", raising=False)

    default_config = TrainingConfig()

    assert TrainingConfig.from_env() == default_config


def test_training_config_from_env_applies_epoch_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCFREC_EPOCHS", "7")

    assert TrainingConfig.from_env().epochs == 7


def test_training_config_from_env_validates_epoch_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCFREC_EPOCHS", "0")

    with pytest.raises(ValueError, match="epochs and batch_size must be positive"):
        TrainingConfig.from_env()
