import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from addict import Dict
from safetensors.numpy import save_file as save_safetensors
from safetensors.numpy import load_file as load_safetensors

from pointspace.datasets.las_image import LasImageDataset
from pointspace.engines.hooks.observation import ObservationCurriculumHook
from pointspace.models.backbone.hpsd import hpsd_v1m1
from pointspace.models.backbone.hpsd.hpsd_v1m1 import (
    HierarchicalPatchSetDistiller,
)
from pointspace.models.backbone.oc_hpsd.oc_hpsd_v1m1 import (
    ObservationConditionedHPSD,
)
from pointspace.models.backbone.oc_hpsd.oc_hpsd_v1m2 import (
    ObservationConditionedHPSDV1M2,
)
from pointspace.models.backbone.oc_hpsd.ops import (
    GeometryGuidedMaskGenerator,
    aggregate_tokens_to_patches_routed,
    build_routed_token_patch_edges,
)
from utils.dino.extract_dino_feature import _update_correspondence_safetensors
from utils.tile_las_image import LASImageTileProcessor


class DummyMaskEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, channels))


class DummyMaskHierarchyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = DummyMaskEmbedding(4)
        self.level0 = nn.Parameter(torch.randn(8, 4))
        self.level1 = nn.Linear(4, 6)
        self.level2 = nn.Linear(6, 8)

    def forward(self, input_dict, return_hierarchy=False):
        mask = input_dict.get(
            "mask", torch.zeros(8, dtype=torch.bool, device=self.level0.device)
        )
        feature0 = torch.where(
            mask.unsqueeze(1),
            self.embedding.mask_token.to(self.level0.dtype),
            self.level0,
        )
        feature1 = self.level1(feature0.reshape(4, 2, 4).mean(dim=1))
        feature2 = self.level2(feature1.reshape(2, 2, 6).mean(dim=1))
        input_to_level = (
            torch.arange(8, device=feature0.device),
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=feature0.device),
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=feature0.device),
        )
        inverse = (
            None,
            input_to_level[1],
            torch.tensor([0, 0, 1, 1], device=feature0.device),
        )
        coords = (
            input_dict["coord"],
            input_dict["coord"][[0, 2, 4, 6]],
            input_dict["coord"][[0, 4]],
        )
        features = (feature0, feature1, feature2)
        batches = (
            torch.zeros(8, dtype=torch.long, device=feature0.device),
            torch.zeros(4, dtype=torch.long, device=feature0.device),
            torch.zeros(2, dtype=torch.long, device=feature0.device),
        )
        hierarchy = []
        for level, feature in enumerate(features):
            point = Dict(feat=feature, coord=coords[level], batch=batches[level])
            if inverse[level] is not None:
                point.pooling_inverse = inverse[level]
            hierarchy.append(
                SimpleNamespace(
                    level=level,
                    point=point,
                    input_to_level=input_to_level[level],
                )
            )
        return hierarchy[-1].point, tuple(hierarchy)


def make_input():
    return dict(
        coord=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 1.5],
                [2.0, 0.0, 0.0],
                [2.2, 0.0, 1.5],
                [4.0, 0.0, 0.0],
                [4.2, 0.0, 1.5],
                [6.0, 0.0, 0.0],
                [6.2, 0.0, 1.5],
            ]
        ),
        offset=torch.tensor([8]),
        dino_feature=torch.randn(4, 10),
        image_patch_index=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        image_valid=torch.ones(8, dtype=torch.bool),
        image_observability=torch.ones(8),
        dino_offset=torch.tensor([4]),
    )


def make_oc_model(monkeypatch):
    backbone = DummyMaskHierarchyBackbone()
    monkeypatch.setattr(hpsd_v1m1, "build_model", lambda config: config)
    return ObservationConditionedHPSD(
        backbone=backbone,
        distill_level=0,
        level_channels=(4, 6, 8),
        teacher_channels=10,
        projector_hidden_channels=12,
        completion_hidden_channels=11,
        mask_rate=0.5,
        lambda_csc=0.2,
        curriculum_start=0.1,
        curriculum_warmup=0.1,
        masking=dict(
            block_size=1.0,
            min_observability=0.6,
            min_vertical_span=1.0,
            min_anchor_points=2,
            min_anchor_ratio=0.5,
            max_mask_points=4,
        ),
    )


def test_routed_edges_keep_anchor_and_masked_statistics():
    edges = build_routed_token_patch_edges(
        input_to_level=torch.tensor([0, 0, 0, 1, 1, 1]),
        patch_index=torch.tensor([0, 0, 1, 1, 1, 2]),
        valid=torch.ones(6, dtype=torch.bool),
        observability=torch.tensor([1.0, 0.5, 1.0, 1.0, 0.5, 1.0]),
        simulated_mask=torch.tensor([False, True, False, True, False, True]),
        num_tokens=2,
        num_patches=3,
        validate_mapping=True,
    )
    assert edges.token.tolist() == [0, 0, 1, 1]
    assert edges.patch.tolist() == [0, 1, 1, 2]
    assert edges.anchor_count.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert edges.masked_count.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert torch.allclose(edges.anchor_q_sum, torch.tensor([1.0, 1.0, 0.5, 0.0]))
    assert torch.allclose(edges.masked_q_sum, torch.tensor([0.5, 0.0, 1.0, 1.0]))

    token_feature = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    patch_feature, used_patch, _ = aggregate_tokens_to_patches_routed(
        token_feature, edges, route="anchor"
    )
    assert used_patch.tolist() == [0, 1]
    assert torch.allclose(patch_feature[0], token_feature[0])
    expected_patch1 = (token_feature[0] * 1.0 + token_feature[1] * 0.5) / 1.5
    assert torch.allclose(patch_feature[1], expected_patch1)


def test_geometry_mask_respects_candidates_and_anchor_budget():
    torch.manual_seed(3)
    data = make_input()
    generator = GeometryGuidedMaskGenerator(
        block_size=1.0,
        min_vertical_span=1.0,
        min_anchor_points=2,
        min_anchor_ratio=0.5,
        max_mask_points=4,
    )
    batch = torch.zeros(8, dtype=torch.long)
    mask = generator(
        data["coord"],
        batch,
        data["image_valid"],
        data["image_observability"],
        mask_rate=0.5,
    )
    assert 0 < int(mask.sum()) <= 4
    assert int((~mask & data["image_valid"]).sum()) >= 4
    assert not generator(
        data["coord"],
        batch,
        data["image_valid"],
        data["image_observability"],
        mask_rate=0.0,
    ).any()


def test_geometry_mask_can_fill_budget_from_one_partial_boundary_block():
    torch.manual_seed(9)
    coord = torch.stack(
        (
            torch.arange(10, dtype=torch.float32) * 0.01,
            torch.zeros(10),
            torch.arange(10, dtype=torch.float32),
        ),
        dim=1,
    )
    generator = GeometryGuidedMaskGenerator(
        block_size=1.0,
        min_vertical_span=1.0,
        min_anchor_points=2,
        min_anchor_ratio=0.5,
        fill_partial_block=True,
    )
    mask, stats = generator(
        coord,
        torch.zeros(10, dtype=torch.long),
        torch.ones(10, dtype=torch.bool),
        torch.ones(10),
        mask_rate=0.5,
        return_stats=True,
    )
    assert stats.requested_count.item() == 5
    assert mask.sum().item() == 5


def test_zero_mask_is_numerically_equal_to_hpsd(monkeypatch):
    torch.manual_seed(4)
    model = make_oc_model(monkeypatch)
    baseline = HierarchicalPatchSetDistiller(
        backbone=copy.deepcopy(model.backbone),
        distill_level=0,
        level_channels=(4, 6, 8),
        teacher_channels=10,
        projector_hidden_channels=12,
    )
    baseline.student_projector.load_state_dict(model.student_projector.state_dict())
    model.set_train_progress(0.0)
    input_dict = make_input()
    oc_result = model(input_dict)
    hpsd_result = baseline(input_dict)
    assert oc_result["mr"].item() == 0.0
    assert oc_result["msk"].item() == 0.0
    assert torch.allclose(oc_result["loss"], hpsd_result["loss"], atol=1e-7)


def test_masked_csc_forward_backward_and_export(monkeypatch):
    torch.manual_seed(5)
    model = make_oc_model(monkeypatch)
    model.set_train_progress(1.0)
    input_dict = make_input()
    result = model(input_dict)
    assert result["mr"].item() == 0.5
    assert result["msk"].item() > 0
    assert result["ctok"].item() > 0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.completion_projector.parameters()
    )
    assert model.backbone.embedding.mask_token.grad is not None

    exported = model(input_dict, return_point_feature=True)
    assert exported["point_feature"].shape == (8, 10)


def test_v1m2_uses_soft_csc_weight_and_reports_effective_rates(monkeypatch):
    torch.manual_seed(5)
    backbone = DummyMaskHierarchyBackbone()
    monkeypatch.setattr(hpsd_v1m1, "build_model", lambda config: config)
    model = ObservationConditionedHPSDV1M2(
        backbone=backbone,
        distill_level=0,
        level_channels=(4, 6, 8),
        teacher_channels=10,
        projector_hidden_channels=12,
        completion_hidden_channels=11,
        mask_rate=0.5,
        lambda_csc=0.2,
        curriculum_start=0.0,
        curriculum_warmup=0.0,
        masking=dict(
            block_size=1.0,
            min_observability=0.6,
            min_vertical_span=1.0,
            min_anchor_points=2,
            min_anchor_ratio=0.5,
            max_mask_points=4,
        ),
    )
    result = model(make_input())
    assert result["msk"].item() == 4
    assert result["mr"].item() == 0.5
    assert result["mu"].item() == 1.0
    assert 0 < result["cr"].item() <= 1
    assert "anc" not in result
    result["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.completion_projector.parameters()
    )


def test_sample_balanced_soft_csc_weights_do_not_favor_large_sample():
    loss = torch.tensor([1.0, 3.0, 10.0])
    weight = torch.tensor([1.0, 3.0, 2.0])
    token = torch.tensor([0, 1, 2])
    token_batch = torch.tensor([0, 0, 1])
    actual = ObservationConditionedHPSD._sample_balanced_weighted_token_mean(
        loss, weight, token, token_batch
    )
    expected = ((1.0 * 1.0 + 3.0 * 3.0) / 4.0 + 10.0) / 2.0
    assert torch.allclose(actual, torch.tensor(expected))


def test_empty_supervision_keeps_both_projectors_in_graph(monkeypatch):
    model = make_oc_model(monkeypatch)
    input_dict = make_input()
    input_dict["image_valid"] = torch.zeros(8, dtype=torch.bool)
    input_dict["image_observability"] = torch.zeros(8)
    input_dict["image_patch_index"] = torch.full((8,), -1, dtype=torch.long)
    result = model(input_dict)
    assert result["loss"].item() == 0.0
    result["loss"].backward()
    assert all(
        parameter.grad is not None for parameter in model.student_projector.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in model.completion_projector.parameters()
    )


def test_mapping_observability_and_legacy_fallback(tmp_path):
    pixel = np.asarray([[0, 0], [1, 1]], dtype=np.int32)
    patch = np.asarray([0, -1], dtype=np.int32)
    valid = np.asarray([True, False])
    new_path = tmp_path / "new.safetensors"
    save_safetensors(
        {
            "pixel_coord": pixel,
            "patch_index": patch,
            "valid": valid,
            "observability": np.asarray([0.75, 0.0], dtype=np.float16),
        },
        new_path,
    )
    _, _, loaded_valid, observability, _ = LasImageDataset._load_mapping(
        new_path, 2
    )
    assert loaded_valid.tolist() == [True, False]
    assert torch.allclose(observability, torch.tensor([0.75, 0.0]))

    old_path = tmp_path / "old.safetensors"
    save_safetensors(
        {"pixel_coord": pixel, "patch_index": patch, "valid": valid}, old_path
    )
    _, _, loaded_valid, observability, _ = LasImageDataset._load_mapping(
        old_path, 2
    )
    assert torch.equal(observability, loaded_valid.float())


def test_surface_state_and_dino_update_preserve_observability(tmp_path):
    processor = LASImageTileProcessor.__new__(LASImageTileProcessor)
    processor.surface_cell_size = 1.0
    processor.surface_radius = 0.0
    processor.surface_z_tolerance = 0.15
    processor.reference_raster = {"resolution": (1.0, 1.0)}
    processor.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    points = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.9], [2.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    visible, observability = processor._compute_surface_state(points)
    assert visible.tolist() == [True, True, True]
    assert observability[0] == 1.0
    assert 0.0 < observability[1] < observability[0]

    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    save_safetensors(
        {
            "pixel_coord": np.asarray([[0, 0], [20, 20]], dtype=np.int32),
            "valid": np.asarray([True, True]),
            "observability": np.asarray([0.75, 0.5], dtype=np.float16),
            "custom": np.asarray([3, 4], dtype=np.int16),
        },
        source,
        metadata={"surface_z_tolerance": "0.15"},
    )
    _update_correspondence_safetensors(
        source,
        output,
        dict(
            original_size=[16, 16],
            patch_size=8,
            feature_shape=[4, 8],
            feature_grid_size=[2, 2],
        ),
    )
    updated = load_safetensors(output)
    assert updated["custom"].tolist() == [3, 4]
    assert updated["observability"].tolist() == [0.75, 0.0]
    assert updated["patch_index"].tolist() == [0, -1]


def test_observation_curriculum_hook_reconstructs_global_progress():
    class ProgressModel:
        def __init__(self):
            self.progress = None

        def set_train_progress(self, progress):
            self.progress = progress

    model = ProgressModel()
    hook = ObservationCurriculumHook()
    hook.trainer = SimpleNamespace(
        model=model,
        train_loader=list(range(5)),
        max_epoch=2,
        epoch=0,
        comm_info={"iter": 0},
    )
    hook.before_train()
    assert model.progress == 0.0
    hook.trainer.epoch = 1
    hook.trainer.comm_info["iter"] = 4
    hook.before_step()
    assert model.progress == 1.0
