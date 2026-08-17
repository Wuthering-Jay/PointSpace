from types import SimpleNamespace

import torch
import torch.nn as nn
from addict import Dict

from pointspace.datasets.transform import CompactDinoPatches
from pointspace.models.backbone.hpsd import hpsd_v1m1
from pointspace.models.backbone.hpsd.hpsd_v1m1 import (
    HierarchicalPatchSetDistiller,
    aggregate_tokens_to_patches,
    build_token_patch_edges,
)


def test_fusion_adapter_and_projector_forward_and_backward(monkeypatch):
    monkeypatch.setattr(hpsd_v1m1, "build_model", lambda config: nn.Identity())

    model = HierarchicalPatchSetDistiller(
        backbone={},
        fusion_levels=(True,),
        fusion_channels=20,
        level_weight_init=(1.0,),
        level_weight_floor=0.0,
        level_channels=(12,),
        teacher_channels=16,
    )
    assert model.active_levels == (0,)
    assert model.fusion_level == 0
    adapted = model.level_adapters["0"](torch.randn(7, 12, requires_grad=True))
    prediction = model.student_projector(adapted)
    assert prediction.shape == (7, 16)
    prediction.square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.level_adapters.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in model.student_projector.parameters()
    )


def test_weighted_hierarchy_fusion_has_single_loss_and_full_gradients(monkeypatch):
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(6, 4)),
                    nn.Parameter(torch.randn(3, 6)),
                    nn.Parameter(torch.randn(2, 8)),
                ]
            )
            self.mappings = (
                torch.arange(6),
                torch.tensor([0, 0, 1, 1, 2, 2]),
                torch.tensor([0, 0, 0, 0, 1, 1]),
            )
            self.pooling_inverse = (
                None,
                torch.tensor([0, 0, 1, 1, 2, 2]),
                torch.tensor([0, 0, 1]),
            )

        def forward(self, input_dict, return_hierarchy=False):
            hierarchy = []
            for level, (feature, mapping) in enumerate(
                zip(self.features, self.mappings)
            ):
                point = Dict(feat=feature)
                if self.pooling_inverse[level] is not None:
                    point.pooling_inverse = self.pooling_inverse[level]
                hierarchy.append(
                    SimpleNamespace(
                        level=level,
                        point=point,
                        input_to_level=mapping,
                    )
                )
            return Dict(), tuple(hierarchy)

    backbone = DummyBackbone()
    monkeypatch.setattr(hpsd_v1m1, "build_model", lambda config: config)
    model = HierarchicalPatchSetDistiller(
        backbone=backbone,
        fusion_levels=(True, True, True),
        fusion_channels=12,
        level_weight_init=(1.0, 0.5, 0.25),
        level_weight_floor=0.05,
        level_channels=(4, 6, 8),
        teacher_channels=10,
    )
    input_dict = dict(
        dino_feature=torch.randn(3, 10),
        dino_patch_index=torch.tensor([0, 0, 1, 1, 2, 2]),
        dino_valid=torch.ones(6, dtype=torch.bool),
        dino_offset=torch.tensor([3]),
    )
    result = model(input_dict)
    assert torch.isfinite(result["loss"])
    weights = model.get_level_weights()
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    assert torch.all(weights >= 0.05)
    assert {"loss", "tok", "edge", "patch"}.issubset(result)
    assert "patch_loss" not in result
    assert all(f"w{level}" in result for level in range(3))
    result["loss"].backward()
    assert all(feature.grad is not None for feature in backbone.features)
    assert all(
        parameter.grad is not None
        for parameter in model.level_adapters.parameters()
    )
    assert model.level_weight_logits.grad is not None
    assert all(
        parameter.grad is not None
        for parameter in model.student_projector.parameters()
    )
    assert model.extract_point_feature(
        input_dict, feature_source="projected"
    ).shape == (6, 10)
    assert model.extract_point_feature(
        input_dict, feature_source="backbone"
    ).shape == (6, 12)
    with torch.no_grad():
        _, hierarchy = model._encode(input_dict)
        reference, fused, _ = model._fuse_hierarchy_features(hierarchy)
        edges = build_token_patch_edges(
            reference.input_to_level,
            input_dict["dino_patch_index"],
            input_dict["dino_valid"],
            num_tokens=fused.shape[0],
            num_patches=input_dict["dino_feature"].shape[0],
        )
        fused_patch, _, _ = aggregate_tokens_to_patches(fused, edges)
        projected_patch, _, _ = aggregate_tokens_to_patches(
            model.student_projector(fused), edges
        )
        # 最终 projector 为线性层，因此训练时先聚合后投影
        # 与测试时先投影 token 再 gather 保持同一映射语义。
        assert torch.allclose(
            model.student_projector(fused_patch), projected_patch, atol=1e-6
        )

    model.zero_grad(set_to_none=True)
    empty_input = dict(input_dict)
    empty_input["dino_patch_index"] = torch.full((6,), -1, dtype=torch.long)
    empty_input["dino_valid"] = torch.zeros(6, dtype=torch.bool)
    empty_result = model(empty_input)
    assert empty_result["loss"].item() == 0.0
    empty_result["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.level_adapters.parameters()
    )
    assert model.level_weight_logits.grad is not None
    assert all(
        parameter.grad is not None
        for parameter in model.student_projector.parameters()
    )


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
