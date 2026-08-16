import torch
import torch.nn as nn
from addict import Dict

from pointspace.datasets.transform import CompactDinoPatches
from pointspace.models.backbone.hpsd import hpsd_v1m1
from pointspace.models.backbone.hpsd.hpsd_v1m1 import (
    HierarchicalPatchSetDistiller,
    aggregate_tokens_to_patches,
    build_token_patch_edges,
    fuse_hierarchy_features,
)


def test_student_projector_supports_mlp_and_linear(monkeypatch):
    monkeypatch.setattr(hpsd_v1m1, "build_model", lambda config: nn.Identity())

    mlp = HierarchicalPatchSetDistiller(
        backbone={},
        distill_level=0,
        level_channels=(12,),
        teacher_channels=16,
        projector_type="mlp",
        projector_hidden_channels=20,
    )
    assert mlp.projector_in_channels == 12
    assert isinstance(mlp.student_projector, nn.Sequential)
    prediction = mlp.student_projector(torch.randn(7, 12, requires_grad=True))
    assert prediction.shape == (7, 16)
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in mlp.student_projector.parameters())
    assert mlp.student_projector(torch.empty(0, 12)).shape == (0, 16)

    linear = HierarchicalPatchSetDistiller(
        backbone={},
        distill_level=0,
        level_channels=(12,),
        teacher_channels=16,
        projector_type="linear",
    )
    assert isinstance(linear.student_projector, nn.Linear)
    assert linear.student_projector(torch.randn(7, 12)).shape == (7, 16)


def test_token_patch_edges_preserve_many_to_many_relations():
    input_to_level = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    patch_index = torch.tensor([0, 0, 1, 1, 2, 2, 3, -1])
    valid = torch.tensor([True, True, True, True, True, True, True, False])

    edges = build_token_patch_edges(
        input_to_level,
        patch_index,
        valid,
        num_tokens=3,
        num_patches=4,
        validate_mapping=True,
    )

    assert edges.token.tolist() == [0, 0, 1, 1, 2, 2]
    assert edges.patch.tolist() == [0, 1, 1, 2, 2, 3]
    assert edges.point_count.tolist() == [2, 1, 1, 1, 1, 1]


def test_patch_aggregation_and_gradient():
    input_to_level = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    patch_index = torch.tensor([0, 0, 1, 1, 2, 2, 3])
    valid = torch.ones(7, dtype=torch.bool)
    edges = build_token_patch_edges(
        input_to_level, patch_index, valid, num_tokens=3, num_patches=4
    )
    token_feat = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]], requires_grad=True
    )

    patch_feat, used_patch, _ = aggregate_tokens_to_patches(
        token_feat, edges, edge_weight="sqrt_count"
    )

    assert used_patch.tolist() == [0, 1, 2, 3]
    assert torch.allclose(patch_feat[0], token_feat[0])
    assert torch.allclose(patch_feat[3], token_feat[2])
    patch_feat.sum().backward()
    assert torch.isfinite(token_feat.grad).all()


def test_empty_relation_is_safe():
    input_to_level = torch.tensor([0, 1])
    patch_index = torch.tensor([-1, -1])
    valid = torch.zeros(2, dtype=torch.bool)
    edges = build_token_patch_edges(
        input_to_level, patch_index, valid, num_tokens=2, num_patches=3
    )
    patch_feat, used_patch, patch_weight = aggregate_tokens_to_patches(
        torch.randn(2, 4), edges
    )
    assert patch_feat.shape == (0, 4)
    assert used_patch.numel() == 0
    assert patch_weight.numel() == 0


def test_sample_balanced_loss():
    # Sample 0 has one patch with loss 1; sample 1 has three patches with loss
    # 3. The correct sample-balanced result is (1 + 3) / 2 = 2 rather than 2.5.
    patch_loss = torch.tensor([1.0, 3.0, 3.0, 3.0])
    patch_index = torch.tensor([0, 1, 2, 3])
    dino_offset = torch.tensor([1, 4])
    loss = HierarchicalPatchSetDistiller._sample_balanced_mean(
        patch_loss, patch_index, dino_offset
    )
    assert torch.allclose(loss, torch.tensor(2.0))


def test_hierarchy_fusion_upcasts_deeper_features():
    class Level:
        def __init__(self, level, feat, inverse=None):
            self.level = level
            self.point = Dict(feat=feat)
            if inverse is not None:
                self.point.pooling_inverse = inverse

    hierarchy = (
        Level(0, torch.zeros(4, 1)),
        Level(1, torch.tensor([[10.0], [20.0]]), torch.tensor([0, 0, 1, 1])),
        Level(2, torch.tensor([[30.0]]), torch.tensor([0, 0])),
    )
    fused = fuse_hierarchy_features(hierarchy, target_level=1)
    assert torch.equal(fused, torch.tensor([[10.0, 30.0], [20.0, 30.0]]))


def test_compact_dino_patches_is_lossless():
    feature = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    patch_index = torch.tensor([4, 1, -1, 4, 3])
    valid = torch.tensor([True, True, False, True, True])
    expected_teacher = feature[patch_index[valid]]
    data = CompactDinoPatches()(
        dict(
            dino_feature=feature,
            dino_patch_index=patch_index,
            dino_valid=valid,
        )
    )
    compact_teacher = data["dino_feature"][data["dino_patch_index"][valid]]
    assert torch.equal(compact_teacher, expected_teacher)
    assert data["dino_feature"].shape == (3, 4)
    assert data["dino_source_patch_index"].tolist() == [1, 3, 4]
    assert data["dino_offset"].tolist() == [3]


def test_compaction_preserves_patch_distillation_loss():
    torch.manual_seed(0)
    token_feat = torch.randn(3, 5)
    teacher = torch.randn(7, 8)
    input_to_level = torch.tensor([0, 0, 1, 1, 2, 2])
    patch_index = torch.tensor([5, 2, 5, 1, 2, -1])
    valid = patch_index >= 0
    projector = torch.nn.Linear(5, 8)

    edges = build_token_patch_edges(
        input_to_level, patch_index, valid, num_tokens=3, num_patches=7
    )
    patch_feat, used_patch, _ = aggregate_tokens_to_patches(token_feat, edges)
    original_loss = 1 - torch.sum(
        torch.nn.functional.normalize(projector(patch_feat), dim=-1)
        * torch.nn.functional.normalize(teacher[used_patch], dim=-1),
        dim=-1,
    ).mean()

    compact = CompactDinoPatches()(
        dict(
            dino_feature=teacher,
            dino_patch_index=patch_index,
            dino_valid=valid,
        )
    )
    compact_edges = build_token_patch_edges(
        input_to_level,
        compact["dino_patch_index"],
        compact["dino_valid"],
        num_tokens=3,
        num_patches=compact["dino_feature"].shape[0],
    )
    compact_feat, compact_patch, _ = aggregate_tokens_to_patches(
        token_feat, compact_edges
    )
    compact_loss = 1 - torch.sum(
        torch.nn.functional.normalize(projector(compact_feat), dim=-1)
        * torch.nn.functional.normalize(
            compact["dino_feature"][compact_patch], dim=-1
        ),
        dim=-1,
    ).mean()
    assert torch.allclose(compact_loss, original_loss)
