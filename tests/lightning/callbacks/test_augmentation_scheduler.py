import pytest
import pytorch_lightning as pl

from fulmo.callbacks import AugmentationOnOffCallback

from ..utils import BoringModel


@pytest.fixture
def tmpdir() -> str:
    """Get temporary directory."""
    return "/tmp"  # noqa: S108


def test_not_enable_augmentation_scheduler(tmpdir: str) -> None:
    """Test the augmentation scheduler."""
    pl_module = BoringModel()
    max_epochs = 5
    scheduler = AugmentationOnOffCallback(max_epochs + 1, -1)
    trainer = pl.Trainer(default_root_dir=tmpdir, max_epochs=max_epochs, callbacks=[scheduler])
    trainer.fit(pl_module)
    assert scheduler.disable


def test_enable_augmentation_scheduler(tmpdir: str) -> None:
    """Test the augmentation scheduler."""
    pl_module = BoringModel()
    max_epochs = 5
    scheduler = AugmentationOnOffCallback(max_epochs - 2, max_epochs + 2)
    trainer = pl.Trainer(default_root_dir=tmpdir, max_epochs=max_epochs, callbacks=[scheduler])
    trainer.fit(pl_module)
    assert not scheduler.disable


def test_disable_augmentation_scheduler(tmpdir: str) -> None:
    """Test the augmentation scheduler."""
    pl_module = BoringModel()
    max_epochs = 5
    scheduler = AugmentationOnOffCallback(max_epochs - 2, max_epochs - 1)
    trainer = pl.Trainer(default_root_dir=tmpdir, max_epochs=max_epochs, callbacks=[scheduler])
    trainer.fit(pl_module)
    assert scheduler.disable
