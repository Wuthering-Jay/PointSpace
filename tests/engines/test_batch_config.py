import pytest

from pointspace.engines.defaults import default_setup
from pointspace.utils.config import Config


def make_config(**kwargs):
    config = dict(
        num_worker=0,
        gradient_accumulation_steps=1,
        seed=42,
    )
    config.update(kwargs)
    return Config(config)


def test_task_specific_batch_config_does_not_require_legacy_or_val_batch():
    cfg = default_setup(
        make_config(
            batch_size_train=4,
            batch_size_test=4,
            gradient_accumulation_steps=4,
        )
    )

    assert cfg.batch_size_train == 4
    assert cfg.batch_size_train_per_gpu == 4
    assert cfg.batch_size_per_gpu == 1
    assert cfg.batch_size_val_per_gpu == 1
    assert cfg.batch_size_test_per_gpu == 4


def test_legacy_batch_size_is_published_as_batch_size_train():
    cfg = default_setup(
        make_config(
            batch_size=8,
            gradient_accumulation_steps=2,
        )
    )

    assert cfg.batch_size_train == 8
    assert cfg.batch_size_per_gpu == 4
    assert cfg.batch_size_val_per_gpu == 1
    assert cfg.batch_size_test_per_gpu == 1


def test_missing_train_batch_size_is_rejected():
    with pytest.raises(ValueError, match="batch_size_train"):
        default_setup(make_config())


def test_non_positive_gradient_accumulation_is_rejected():
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        default_setup(
            make_config(
                batch_size_train=4,
                gradient_accumulation_steps=0,
            )
        )

