import pytest
import pytorch_lightning as pl
import torch

from fulmo.callbacks import ScheduledDropoutCallback

from ..utils import BoringModel


@pytest.fixture
def tmpdir() -> str:
    """Get temporary directory."""
    return "/tmp"  # noqa: S108


@pytest.mark.parametrize(
    "layer_name",
    [
        "backbone",
        "head",
    ],
)
def test_dropout_scheduler(tmpdir: str, layer_name: str) -> None:
    """Test that the dropout scheduler changes probabilities"""
    pl_module = BoringModel()
    max_epochs = 5
    scheduler = ScheduledDropoutCallback(max_epochs, layer_name)
    initial_probabilities = [
        layer.p for layer in getattr(pl_module.model, layer_name).children() if isinstance(layer, torch.nn.Dropout)
    ]

    trainer = pl.Trainer(default_root_dir=tmpdir, max_epochs=max_epochs, callbacks=[scheduler])
    trainer.fit(pl_module)
    probabilities_after_train = [
        layer.p for layer in getattr(pl_module.model, layer_name).children() if isinstance(layer, torch.nn.Dropout)
    ]
    assert all([before > after for before, after in zip(initial_probabilities, probabilities_after_train)])
