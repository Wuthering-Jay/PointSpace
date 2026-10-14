from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from pointspace.engines.hooks.misc import (
    FreezeBatchNormStatsHook,
    FreezeStochasticModulesHook,
    GradualUnfreezeHook,
    PreciseBatchNormHook,
    _BatchNormMoments,
)




def test_exact_batch_norm_moments_are_point_weighted():
    bn = nn.BatchNorm1d(2)
    bn.train()
    moments = _BatchNormMoments([("bn", bn)])
    samples = [
        torch.tensor([[1.0, 10.0], [3.0, 30.0]]),
        torch.tensor([[5.0, 50.0], [7.0, 70.0], [9.0, 90.0]]),
    ]
    for sample in samples:
        bn(sample)
    moments.finalize()
    moments.close()
    expected = torch.cat(samples)
    assert torch.allclose(bn.running_mean, expected.mean(0))
    assert torch.allclose(bn.running_var, expected.var(0, unbiased=True))


def test_precise_bn_decoder_scope_excludes_encoder():
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.embedding = nn.Sequential(nn.BatchNorm1d(3))
    model.backbone.enc = nn.Sequential(nn.BatchNorm1d(3))
    model.backbone.dec = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 3))
    hook = PreciseBatchNormHook(data={}, scope="decoder")
    selected = hook._selected_modules(model)
    assert tuple(selected) == ("backbone.dec.0",)


def test_gradual_unfreeze_freezes_then_releases_only_encoder():
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.embedding = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    model.backbone.enc = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    model.backbone.dec = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    trainer = SimpleNamespace(
        model=model, start_epoch=0, epoch=0, logger=Mock()
    )
    hook = GradualUnfreezeHook(
        ("backbone.embedding", "backbone.enc"), frozen_epochs=1
    )
    hook.trainer = trainer
    hook.before_train()
    assert not any(p.requires_grad for p in model.backbone.embedding.parameters())
    assert not any(p.requires_grad for p in model.backbone.enc.parameters())
    assert all(p.requires_grad for p in model.backbone.dec.parameters())
    assert not model.backbone.embedding.training
    assert not model.backbone.enc.training

    trainer.epoch = 1
    model.train()
    hook.before_epoch()
    assert all(p.requires_grad for p in model.backbone.embedding.parameters())
    assert all(p.requires_grad for p in model.backbone.enc.parameters())
    assert model.backbone.embedding.training
    assert model.backbone.enc.training


def test_freeze_batch_norm_stats_keeps_affine_trainable():
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.embedding = nn.Sequential(nn.BatchNorm1d(3))
    model.backbone.enc = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    model.backbone.dec = nn.Sequential(nn.BatchNorm1d(3))
    hook = FreezeBatchNormStatsHook(
        module_prefixes=("backbone.embedding", "backbone.enc")
    )
    hook.trainer = SimpleNamespace(model=model, logger=Mock())

    model.train()
    hook.before_train()
    assert not model.backbone.embedding[0].training
    assert not model.backbone.enc[1].training
    assert model.backbone.dec[0].training
    assert model.backbone.embedding[0].weight.requires_grad

    model.train()
    hook.before_epoch()
    assert not model.backbone.embedding[0].training
    assert not model.backbone.enc[1].training
    assert model.backbone.dec[0].training


def test_freeze_stochastic_modules_keeps_encoder_and_bn_trainable():
    class DropPath(nn.Module):
        def forward(self, value):
            return value

    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.embedding = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(0.2))
    model.backbone.enc = nn.Sequential(nn.Linear(3, 3), DropPath())
    model.backbone.dec = nn.Sequential(nn.Dropout(0.2))
    hook = FreezeStochasticModulesHook(
        module_prefixes=("backbone.embedding", "backbone.enc")
    )
    hook.trainer = SimpleNamespace(model=model, logger=Mock())

    model.train()
    hook.before_train()
    assert model.backbone.embedding.training
    assert model.backbone.embedding[0].training
    assert not model.backbone.embedding[1].training
    assert model.backbone.enc.training
    assert not model.backbone.enc[1].training
    assert model.backbone.dec[0].training
    assert all(parameter.requires_grad for parameter in model.parameters())

    model.train()
    hook.before_epoch()
    assert model.backbone.embedding[0].training
    assert not model.backbone.embedding[1].training
    assert not model.backbone.enc[1].training
    assert model.backbone.dec[0].training

