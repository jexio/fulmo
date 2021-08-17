from typing import List

import albumentations as A
import pandas as pd
import pytest
from torch.utils.data import DataLoader, Dataset

from fulmo.datasets import MultiDomainCsvDataset
from fulmo.readers import Augmentor, ReaderCompose

from ..utils import NpyGenerator


@pytest.fixture
def reader() -> ReaderCompose:
    """Create a new instance of `ReaderCompose`"""
    open_fn = ReaderCompose(
        [
            NpyGenerator(output_key="feature_1", shape=(256, 256, 3)),
            NpyGenerator(output_key="feature_2", shape=(512, 512, 3)),
        ]
    )
    return open_fn


@pytest.fixture
def x1_transforms() -> Augmentor:
    """Create a new instance of `Augmentor`"""
    train = Augmentor("feature_1", "transformed_feature_1", augment_fn=lambda x: x)
    return train


@pytest.fixture
def x2_transforms() -> Augmentor:
    """Create a new instance of `Augmentor`"""
    compose_transforms = A.Compose([A.Resize(128, 128, always_apply=True, p=1.0)])
    train = Augmentor("feature_2", "transformed_feature_2", augment_fn=lambda x: compose_transforms(image=x)["image"])
    return train


@pytest.fixture
def dataset(reader: ReaderCompose, x1_transforms: Augmentor, x2_transforms: Augmentor) -> Dataset:
    """Create a new instance of `Dataset`"""
    train_df = pd.DataFrame([{"feature_1": "", "feature_2": ""}] * 32)
    transforms = {"feature_1": x1_transforms, "feature_2": x2_transforms}
    data_train = MultiDomainCsvDataset(train_df, reader, transforms)
    return data_train


def _check(keys: List[str], keys_from_batch: List[str]) -> bool:
    """Check that each key contains in `keys_from_batch`"""
    if len(keys) != len(keys_from_batch):
        return False
    return all([True if key in keys_from_batch else False for key in keys])


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        4,
    ],
)
def test_multi_domain_dataset(dataset: Dataset, batch_size: int) -> None:
    """Test that the `MultiDomainCsvDataset` works"""
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    items = list(dataloader)
    keys = ["transformed_feature_1", "transformed_feature_2", "feature_1", "feature_2"]
    assert all([_check(keys, list(item.keys())) for item in items])
    assert all([item["transformed_feature_1"].shape == (batch_size, 256, 256, 3) for item in items])
    assert all([item["transformed_feature_2"].shape == (batch_size, 128, 128, 3) for item in items])
    assert all([item["feature_2"].shape == (batch_size, 512, 512, 3) for item in items])
