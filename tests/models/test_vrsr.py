from types import SimpleNamespace

import torch
import torch.nn as nn
from addict import Dict

from pointspace.models.backbone.hpsd import hpsd_v1m1
from pointspace.models.backbone.hpsd.hpsd_v1m1 import (
    HPSDTrainContext,
    HierarchicalPatchSetDistiller,
    TokenPatchEdges,
)
from pointspace.models.backbone.vrsr.ops import (
    aggregate_patch_teacher_to_tokens,
    chunked_topk_cosine,
    compute_token_visibility,
)
from pointspace.models.backbone.vrsr.vrsr_v1m1 import (
    HPSDVRSRDistiller,
    VisibilityReliableSupervisor,
)


class DummyHierarchyBackbone(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.feature = nn.Parameter(torch.randn(4, channels))

    def forward(self, input_dict, return_hierarchy=False):
        point = Dict(
            feat=self.feature,
            coord=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]
            ),
            batch=torch.tensor([0, 0, 0, 0]),
        )
        level = SimpleNamespace(
            level=0,
            point=point,
            input_to_level=torch.arange(4),
        )
        return point, (level,)


def test_token_visibility_and_teacher_aggregation():
    mapping = torch.tensor([0, 0, 1, 1, 2, 2])
    valid = torch.tensor([True, False, True, True, False, False])
    visibility, valid_count, total_count = compute_token_visibility(mapping, valid, 3)
    assert torch.equal(valid_count, torch.tensor([1.0, 2.0, 0.0]))
    assert torch.equal(total_count, torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(visibility, torch.tensor([0.5, 1.0, 0.0]))

    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    edges = TokenPatchEdges(
        token=torch.tensor([0, 0, 1]),
        patch=torch.tensor([0, 1, 2]),
        point_count=torch.tensor([1, 1, 4]),
    )
    stats = aggregate_patch_teacher_to_tokens(teacher, edges)
    assert stats.token.tolist() == [0, 1]
    assert stats.feature.shape == (2, 2)
    assert stats.point_count.tolist() == [2.0, 4.0]
    assert stats.patch_count.tolist() == [2.0, 1.0]
    assert stats.purity[1].item() == 1.0
    assert 0.7 < stats.purity[0].item() < 0.8


def test_chunked_topk_matches_dense_result():
    torch.manual_seed(1)
    query = torch.nn.functional.normalize(torch.randn(17, 8), dim=-1)
    key = torch.nn.functional.normalize(torch.randn(11, 8), dim=-1)
    value, index = chunked_topk_cosine(query, key, topk=4, chunk_size=3)
    expected_value, expected_index = torch.topk(query @ key.T, k=4, dim=1)
    assert torch.allclose(value, expected_value)
    assert torch.equal(index, expected_index)


def test_hpsd_context_is_numerically_lossless(monkeypatch):
    backbone = DummyHierarchyBackbone(channels=6)
    monkeypatch.setattr(hpsd_v1m1, "build_model", lambda config: config)
    model = HierarchicalPatchSetDistiller(
        backbone=backbone,
        distill_level=0,
        level_channels=(6,),
        teacher_channels=8,
        projector_hidden_channels=10,
    )
    input_dict = dict(
        dino_feature=torch.randn(2, 8),
        dino_patch_index=torch.tensor([0, 1, -1, -1]),
        dino_valid=torch.tensor([True, True, False, False]),
        dino_offset=torch.tensor([2]),
    )
    normal = model(input_dict)
    contextual, context = model.forward_train(input_dict, return_context=True)
    assert torch.equal(normal["loss"], contextual["loss"])
    assert normal.keys() == contextual.keys()
    assert isinstance(context, HPSDTrainContext)
    # context 只引用 HPSD 本次前向已经生成的 distill_feat；单层 concat 本身
    # 由原实现创建新 tensor，因此这里只验证数值和计算图仍连接 backbone。
    assert torch.equal(context.distill_feat, backbone.feature)
    assert context.distill_feat.grad_fn is not None
    assert context.teacher.shape == (2, 8)


def test_local_loss_has_target_gradient_but_detaches_source():
    torch.manual_seed(2)
    feature = torch.randn(4, 6, requires_grad=True)
    point = Dict(
        feat=feature,
        coord=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]
        ),
        batch=torch.zeros(4, dtype=torch.long),
    )
    level = SimpleNamespace(point=point, input_to_level=torch.arange(4), level=0)
    edges = TokenPatchEdges(
        token=torch.tensor([0, 1]),
        patch=torch.tensor([0, 1]),
        point_count=torch.tensor([4, 4]),
    )
    context = HPSDTrainContext(
        point=point,
        hierarchy=(level,),
        level=level,
        distill_feat=feature,
        edges=edges,
        teacher=torch.nn.functional.normalize(torch.randn(2, 8), dim=-1),
    )
    module = VisibilityReliableSupervisor(
        in_channels=6,
        teacher_channels=8,
        propagation_channels=4,
        hidden_channels=7,
        mode="local",
        source_q=1.0,
        target_q=0.0,
        min_source_points=1,
        topk=2,
        lambda_cal=0.0,
        lambda_local=1.0,
    )
    result = module(
        context,
        dict(dino_valid=torch.tensor([True, True, False, False])),
    )
    assert result["source_count"].item() == 2
    assert result["target_count"].item() == 2
    assert result["accepted_count"].item() == 2
    assert torch.isfinite(result["local"])
    result["local"].backward()
    assert torch.count_nonzero(feature.grad[2:]).item() > 0
    assert torch.count_nonzero(feature.grad[:2]).item() == 0


def test_calibration_mode_and_empty_local_are_safe():
    feature = torch.randn(2, 5, requires_grad=True)
    point = Dict(
        feat=feature,
        coord=torch.zeros(2, 3),
        batch=torch.zeros(2, dtype=torch.long),
    )
    level = SimpleNamespace(point=point, input_to_level=torch.arange(2), level=0)
    context = HPSDTrainContext(
        point=point,
        hierarchy=(level,),
        level=level,
        distill_feat=feature,
        edges=TokenPatchEdges(
            token=torch.tensor([0, 1]),
            patch=torch.tensor([0, 1]),
            point_count=torch.tensor([4, 4]),
        ),
        teacher=torch.randn(2, 8),
    )
    module = VisibilityReliableSupervisor(
        in_channels=5,
        teacher_channels=8,
        propagation_channels=4,
        mode="calibrate",
        source_q=1.0,
        min_source_points=1,
    )
    result = module(context, dict(dino_valid=torch.ones(2, dtype=torch.bool)))
    assert result["target_count"].item() == 0
    assert result["local"].item() == 0.0
    result["loss"].backward()
    assert torch.count_nonzero(feature.grad).item() > 0


def test_local_retrieval_never_crosses_batch_samples():
    module = VisibilityReliableSupervisor(
        in_channels=2,
        teacher_channels=4,
        propagation_channels=2,
        hidden_channels=3,
        mode="local",
        topk=1,
        temperature=0.1,
    )
    # 每个样本的 target 都与另一个样本的 source 相同、与本样本 source 相反。
    # 若检索错误跨样本，loss 会为 0；正确按样本分组时 loss 应为 2。
    student = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
        requires_grad=True,
    )
    loss, accepted, _ = module._local_loss(
        student=student,
        source_token=torch.tensor([0, 2]),
        target_token=torch.tensor([1, 3]),
        token_batch=torch.tensor([0, 0, 1, 1]),
        coord=torch.zeros(4, 3),
    )
    assert accepted == 2
    assert torch.allclose(loss, torch.tensor(2.0))


def test_empty_supervision_keeps_zero_gradient_graph():
    feature = torch.randn(3, 5, requires_grad=True)
    point = Dict(
        feat=feature,
        coord=torch.zeros(3, 3),
        batch=torch.zeros(3, dtype=torch.long),
    )
    level = SimpleNamespace(point=point, input_to_level=torch.arange(3), level=0)
    empty = torch.empty(0, dtype=torch.long)
    context = HPSDTrainContext(
        point=point,
        hierarchy=(level,),
        level=level,
        distill_feat=feature,
        edges=TokenPatchEdges(token=empty, patch=empty, point_count=empty),
        teacher=torch.empty(0, 8),
    )
    module = VisibilityReliableSupervisor(
        in_channels=5,
        teacher_channels=8,
        propagation_channels=4,
        mode="local",
    )
    result = module(context, dict(dino_valid=torch.zeros(3, dtype=torch.bool)))
    assert result["loss"].item() == 0.0
    result["loss"].backward()
    assert feature.grad is not None
    assert torch.count_nonzero(feature.grad).item() == 0


def test_hpsd_vrsr_preserves_export_path(monkeypatch):
    backbone = DummyHierarchyBackbone(channels=6)
    monkeypatch.setattr(hpsd_v1m1, "build_model", lambda config: config)
    model = HPSDVRSRDistiller(
        backbone=backbone,
        distill_level=0,
        level_channels=(6,),
        teacher_channels=8,
        projector_hidden_channels=10,
        vrsr=dict(
            propagation_channels=4,
            hidden_channels=7,
            source_q=1.0,
            min_source_points=1,
            topk=1,
        ),
    )
    input_dict = dict(
        dino_feature=torch.randn(2, 8),
        dino_patch_index=torch.tensor([0, 1, -1, -1]),
        dino_valid=torch.tensor([True, True, False, False]),
        dino_offset=torch.tensor([2]),
    )
    result = model(input_dict)
    assert {"loss", "hpsd", "cal", "loc", "src", "tgt", "acc"}.issubset(result)
    assert torch.isfinite(result["loss"])
    exported = model(input_dict, return_point_feature=True)
    expected = model.extract_point_feature(input_dict)
    assert torch.equal(exported["point_feature"], expected)
