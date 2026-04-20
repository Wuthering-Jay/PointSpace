import importlib.util
import math
import runpy
import sys
from pathlib import Path

import pytest
import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointspace.datasets.transform import (
    Collect,
    GridSampling3D,
    GroundElevation,
    PointFeatures,
    SaveNodeIndex,
)
from pointspace.models.builder import build_model
from pointspace.models.backbone.ezsp.graph_partition import GreedyContourPriorPartition
from pointspace.models.backbone.ezsp.hierarchy_graph_transform import HierarchyGraphTransform
from pointspace.models.backbone.ezsp.sparse_cnn import SparseCNN
from pointspace.models.backbone.ezsp.spt.attention import SelfAttentionBlock
from pointspace.models.backbone.ezsp.spt.pool import AttentivePool
from pointspace.models.backbone.ezsp.spt.stage import PointStage
from pointspace.models.backbone.ezsp.superpoint_hierarchy import SuperpointHierarchy
from pointspace.models.losses.misc import WeightedFocalLoss
from pointspace.models.backbone.ezsp.graph_partition import reverse_horizontal_edge_attr


def _load_reference_weighted_focal():
    ref_path = (
        ROOT
        / "reference_code"
        / "superpoint_transformer"
        / "src"
        / "loss"
        / "focal.py"
    )
    spec = importlib.util.spec_from_file_location("spt_reference_focal", ref_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.WeightedFocalLoss


def test_weighted_focal_matches_reference():
    ref_cls = _load_reference_weighted_focal()
    weight = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    logits = torch.tensor(
        [[2.0, -1.0, 0.2], [0.1, 1.4, -0.3], [0.4, -0.5, 2.2]],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 1, 2], dtype=torch.long)

    ref_loss = ref_cls(weight=weight.clone(), gamma=2.0, ignore_index=3)(logits, target)
    local_loss = WeightedFocalLoss(
        weight=weight.clone(), gamma=2.0, ignore_index=3
    )(logits, target)

    assert torch.allclose(local_loss, ref_loss, atol=1e-6, rtol=1e-6)


def test_sparse_cnn_supports_official_kernel_schedule():
    model = SparseCNN(
        in_channels=5,
        channels=[32, 32, 32],
        kernel_size=[7, 3, 3],
        dilation=[1, 1, 1],
        residual=False,
    )
    assert model.kernel_schedule == [7, 3, 3]
    assert model.dilation_schedule == [1, 1, 1]


def test_horizontal_edge_features_are_aggregated_from_child_graph():
    partitioner = GreedyContourPriorPartition(build_edge_features=True)
    pos_child = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    child_edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 1, 3, 2, 3],
            [1, 0, 2, 0, 3, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    super_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    merged_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    pos_parent = torch.tensor([[0.0, 0.5, 0.0], [2.0, 0.5, 0.0]], dtype=torch.float32)
    normal_parent = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    zeros = torch.zeros(2, dtype=torch.float32)

    edge_attr = partitioner._compute_horizontal_edge_attr_from_child_graph(
        pos_child=pos_child,
        child_edge_index=child_edge_index,
        super_index=super_index,
        merged_edge_index=merged_edge_index,
        pos_parent=pos_parent,
        normal_parent=normal_parent,
        log_length_parent=zeros,
        log_surface_parent=zeros,
        log_volume_parent=zeros,
        log_size_parent=zeros,
    )

    expected_mean_off = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
    expected_mean_dist = math.sqrt(2.0)
    assert torch.allclose(edge_attr[0, 0:3], expected_mean_off, atol=1e-6)
    assert torch.allclose(edge_attr[0, 3:6], torch.zeros(3), atol=1e-6)
    assert torch.allclose(edge_attr[0, 6], torch.tensor(expected_mean_dist), atol=1e-6)


def test_trimmed_horizontal_graph_is_expanded_bidirectionally():
    partitioner = GreedyContourPriorPartition(build_edge_features=True)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.5, 0.4, 0.5, 0.6, 0.7, 0.8, 2.0, 0.9, 1.1, 1.2, 1.3, 1.4],
            [2.0, 0.0, -1.0, 0.2, 0.3, 0.4, 1.7, 0.6, 0.7, -0.2, 0.1, 0.5, 3.0, 0.8, -0.4, 0.9, -1.1, 0.2],
        ],
        dtype=torch.float32,
    )
    bi_edge_index, bi_edge_attr = partitioner._make_horizontal_graph_bidirectional(edge_index, edge_attr)

    assert bi_edge_index.shape == (2, 4)
    assert bi_edge_attr.shape == (4, 18)
    assert torch.equal(bi_edge_index[:, 0], torch.tensor([0, 1]))
    assert torch.equal(bi_edge_index[:, 1], torch.tensor([1, 0]))
    assert torch.allclose(bi_edge_attr[1], reverse_horizontal_edge_attr(bi_edge_attr[0].unsqueeze(0)).squeeze(0))
    assert torch.equal(bi_edge_index[:, 2], torch.tensor([1, 2]))
    assert torch.equal(bi_edge_index[:, 3], torch.tensor([2, 1]))
    assert torch.allclose(bi_edge_attr[3], reverse_horizontal_edge_attr(bi_edge_attr[2].unsqueeze(0)).squeeze(0))


def test_point_stage_supports_point_mlp_on_cnn_feats():
    stage = PointStage(
        in_mlp=[8, 12],
        cnn_blocks=True,
        point_mlp_on_cnn_feats=True,
        use_pos=False,
    )
    x = torch.randn(6, 5)
    x_mlp = torch.randn(6, 3)
    norm_index = torch.zeros(6, dtype=torch.long)

    out, diameter_parent = stage(x, norm_index, x_mlp=x_mlp)

    assert out.shape == (6, 12)
    assert diameter_parent is None


def test_official_dales_configs_are_selected():
    stage1 = runpy.run_path(ROOT / "configs" / "dales" / "ezsp" / "stage1_partition.py")
    stage2 = runpy.run_path(ROOT / "configs" / "dales" / "ezsp" / "stage2_semantic.py")

    assert stage1["epoch"] >= 1
    assert stage1["optimizer"]["lr"] == pytest.approx(5e-4)
    assert stage1["partition_criterion_config"]["adaptive_sampling_ratio"] == pytest.approx(0.7)
    assert stage1["partition_config"]["min_size"] == [10]
    assert stage1["feature_keys"] == ["intensity"]
    assert stage1["in_channels"] == 1
    assert stage1["sparse_cnn_config"]["kernel_size"] == [7, 3, 3]
    assert stage1["data"]["val"]["loop"] == 1
    assert [t["type"] for t in stage1["data"]["val"]["transform"]] == [
        "ZPercentileCenterShift",
        "SaveNodeIndex",
        "Copy",
        "GridSampling3D",
    ]

    assert stage2["epoch"] == 600
    assert stage2["optimizer"]["lr"] == pytest.approx(5e-3)
    assert stage2["partition_config"]["min_size"] == [5, 15, 70]
    assert stage2["transformer_config"]["nano"] is False
    assert stage2["transformer_config"]["point_cnn_blocks"] is True
    assert stage2["transformer_config"]["point_mlp_on_cnn_feats"] is True
    assert stage2["feature_keys"] == ["intensity"]
    assert stage2["partition_feature_keys"] == ["intensity"]
    assert stage2["point_hf_keys"] == ["intensity"]
    assert stage2["in_channels"] == 1
    assert stage2["transformer_config"]["point_mlp"] == [41, 64, 128]
    assert stage2["transformer_config"]["use_node_hf"] is False
    assert stage2["transformer_config"]["pool"] == "max"
    assert stage2["model"]["post_cnn_keys"] == [
        "intensity",
        "linearity",
        "planarity",
        "scattering",
        "verticality",
        "elevation",
    ]
    assert stage2["post_cnn_point_hf_keys"] == stage2["model"]["post_cnn_keys"]
    assert stage2["graph_transform_config"]["type"] == "HierarchyGraphTransform"
    assert stage2["graph_transform_config"]["add_self_loops"] is True
    assert stage2["model"]["criteria"][0]["type"] == "WeightedFocalLoss"


def test_save_node_index_grid_sampling_and_collect_preserve_ezsp_contract():
    data_dict = {
        "coord": np.array(
            [
                [0.00, 0.00, 0.00],
                [0.10, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [1.10, 0.00, 0.00],
            ],
            dtype=np.float32,
        ),
        "segment": np.array([1, 1, 2, 2], dtype=np.int64),
        "echo": np.array([[0.5], [1.5], [2.5], [3.5]], dtype=np.float32),
    }

    data_dict = SaveNodeIndex(key="sub")(data_dict)
    assert data_dict["sub"].tolist() == [0, 1, 2, 3]
    assert "sub" in data_dict["index_valid_keys"]

    data_dict = GridSampling3D(
        size=0.5,
        mode="mean",
        hist_key="segment",
        hist_size=4,
        quantize_coords=False,
        feat_keys=["coord", "echo"],
    )(data_dict)

    assert data_dict["num_raw_points"] == 4
    assert data_dict["coord"].shape == (2, 3)
    assert data_dict["segment"].shape == (2, 4)
    assert data_dict["sub"]["pointer"].tolist() == [0, 2, 4]
    voxel_members = [
        set(data_dict["sub"]["value"][data_dict["sub"]["pointer"][i] : data_dict["sub"]["pointer"][i + 1]].tolist())
        for i in range(2)
    ]
    assert {frozenset(members) for members in voxel_members} == {frozenset({0, 1}), frozenset({2, 3})}
    assert np.allclose(data_dict["feat"][:, :3], data_dict["coord"])
    assert sorted(data_dict["feat"][:, 3].tolist()) == pytest.approx([1.0, 3.0])
    assert {tuple(np.where(data_dict["voxel_inverse"] == i)[0].tolist()) for i in range(2)} == {(0, 1), (2, 3)}
    assert data_dict["grid_size"] == pytest.approx(0.5)

    tensor_dict = {
        "coord": torch.from_numpy(data_dict["coord"]),
        "segment": torch.from_numpy(data_dict["segment"]),
        "sub": {
            "pointer": torch.from_numpy(data_dict["sub"]["pointer"]),
            "value": torch.from_numpy(data_dict["sub"]["value"]),
        },
        "num_raw_points": data_dict["num_raw_points"],
        "grid_size": data_dict["grid_size"],
        "echo": torch.from_numpy(np.array([[1.0], [3.0]], dtype=np.float32)),
    }
    collected = Collect(
        keys=["coord", "segment", "sub", "num_raw_points", "grid_size"],
        feat_keys=["coord", "echo"],
    )(tensor_dict)

    assert collected["coord"].shape == (2, 3)
    assert collected["segment"].shape == (2, 4)
    assert collected["feat"].shape == (2, 4)
    assert sorted(collected["feat"][:, 3].tolist()) == pytest.approx([1.0, 3.0])
    assert collected["offset"].tolist() == [2]


def test_point_features_compute_geometric_descriptors():
    coord = np.stack(
        [
            np.linspace(0.0, 4.0, 32, dtype=np.float32),
            np.zeros(32, dtype=np.float32),
            np.zeros(32, dtype=np.float32),
        ],
        axis=1,
    )
    data_dict = {"coord": coord}
    data_dict = PointFeatures(
        keys=["linearity", "planarity", "scattering", "verticality", "normal"],
        k=8,
        k_min=5,
    )(data_dict)

    assert data_dict["linearity"].shape == (32, 1)
    assert data_dict["normal"].shape == (32, 3)
    assert float(np.median(data_dict["linearity"])) > 0.8
    assert float(np.median(data_dict["planarity"])) < 0.2
    assert float(np.median(data_dict["scattering"])) < 0.2
    assert np.all(np.isfinite(data_dict["verticality"]))


def test_ground_elevation_estimates_plane_relative_height():
    x = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    zz = 0.2 * xx + 0.1 * yy + 1.0
    coord = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1).astype(np.float32)
    outlier = np.array([[0.0, 0.0, 4.0]], dtype=np.float32)
    coord = np.concatenate([coord, outlier], axis=0)

    data_dict = {"coord": coord}
    data_dict = GroundElevation(model="ransac", xy_grid=0.5, scale=1.0)(data_dict)
    elevation = data_dict["elevation"].reshape(-1)

    assert elevation.shape[0] == coord.shape[0]
    assert np.max(np.abs(elevation[:-1])) < 0.25
    assert elevation[-1] > 2.0


def test_vertical_edge_attr_matches_official_field_order():
    partitioner = GreedyContourPriorPartition(build_vertical_features=True)
    pos_child = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    pos_parent = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    super_index = torch.tensor([0, 0], dtype=torch.long)
    normal_child = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    normal_parent = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    log_length_child = torch.tensor([0.1, 0.2], dtype=torch.float32)
    log_surface_child = torch.tensor([0.3, 0.4], dtype=torch.float32)
    log_volume_child = torch.tensor([0.5, 0.6], dtype=torch.float32)
    log_size_child = torch.tensor([0.7, 0.8], dtype=torch.float32)
    log_length_parent = torch.tensor([1.1], dtype=torch.float32)
    log_surface_parent = torch.tensor([1.3], dtype=torch.float32)
    log_volume_parent = torch.tensor([1.5], dtype=torch.float32)
    log_size_parent = torch.tensor([1.7], dtype=torch.float32)

    v_edge_attr = partitioner._compute_vertical_edge_attr(
        pos_child=pos_child,
        pos_parent=pos_parent,
        super_index=super_index,
        node_size_child=torch.ones(2, dtype=torch.long),
        node_size_parent=torch.tensor([2], dtype=torch.long),
        normal_child=normal_child,
        normal_parent=normal_parent,
        log_length_child=log_length_child,
        log_length_parent=log_length_parent,
        log_surface_child=log_surface_child,
        log_surface_parent=log_surface_parent,
        log_volume_child=log_volume_child,
        log_volume_parent=log_volume_parent,
        log_size_child=log_size_child,
        log_size_parent=log_size_parent,
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.9],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(v_edge_attr, expected, atol=1e-6)


def test_hierarchy_graph_transform_restricts_edges_and_adds_self_loops():
    hierarchy = SuperpointHierarchy(
        [
            {
                "pos": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                "x": torch.randn(4, 8),
                "super_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                "v_edge_attr": torch.randn(4, 9),
                "batch": torch.zeros(4, dtype=torch.long),
            },
            {
                "pos": torch.tensor([[0.25, 0.0, 0.0], [2.25, 0.0, 0.0]], dtype=torch.float32),
                "x": torch.randn(2, 16),
                "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "edge_attr": torch.randn(2, 18),
                "batch": torch.zeros(2, dtype=torch.long),
            },
        ]
    )
    transform = HierarchyGraphTransform(
        enabled=True,
        training_only=False,
        apply_levels="1+",
        max_edges=1,
        add_self_loops=True,
        pos_jitter_std=0.01,
        edge_attr_jitter_std=0.01,
    )

    out = transform(hierarchy)

    assert out[1]["edge_index"].shape[1] == 3
    assert torch.equal(out[1]["edge_index"][:, -2], torch.tensor([0, 0]))
    assert torch.equal(out[1]["edge_index"][:, -1], torch.tensor([1, 1]))
    assert torch.isfinite(out[1]["edge_attr"]).all()
    assert torch.isfinite(out[1]["pos"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP regression test")
def test_amp_attention_and_pooling_remain_finite():
    device = torch.device("cuda")
    x = torch.randn(32, 64, device=device, dtype=torch.float16) * 20
    edge_index = torch.stack(
        [
            torch.arange(32, device=device).repeat_interleave(4),
            torch.randint(0, 32, (128,), device=device),
        ],
        dim=0,
    )
    edge_attr = torch.randn(128, 18, device=device, dtype=torch.float16) * 10

    attn = SelfAttentionBlock(
        dim=64,
        num_heads=4,
        qk_dim=16,
        k_rpe=True,
        q_rpe=True,
        in_rpe_dim=18,
    ).to(device)
    pool = AttentivePool(
        dim=64,
        q_in_dim=64,
        num_heads=4,
        in_dim=64,
        qk_dim=16,
        in_rpe_dim=9,
        k_rpe=True,
        q_rpe=True,
    ).to(device)
    x_parent = torch.randn(8, 64, device=device, dtype=torch.float16) * 20
    pool_index = torch.randint(0, 8, (32,), device=device)
    v_edge_attr = torch.randn(32, 9, device=device, dtype=torch.float16) * 10

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out_attn = attn(x, edge_index=edge_index, edge_attr=edge_attr)
        out_pool = pool(x, x_parent, pool_index, edge_attr=v_edge_attr, num_pool=8)

    assert torch.isfinite(out_attn).all()
    assert torch.isfinite(out_pool).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for stage2 forward regression test")
def test_stage2_officialish_forward_remains_finite():
    stage2 = runpy.run_path(ROOT / "configs" / "dales" / "ezsp" / "stage2_semantic.py")
    model = build_model(stage2["model"]).cuda().eval()

    num_voxels = 32
    coord = torch.randn(num_voxels, 3, device="cuda")
    feat = torch.randn(num_voxels, stage2["in_channels"], device="cuda")
    offset = torch.tensor([num_voxels], device="cuda", dtype=torch.long)
    labels = torch.randint(0, stage2["num_classes"], (num_voxels,), device="cuda")
    segment = torch.nn.functional.one_hot(
        labels, num_classes=stage2["num_classes"] + 1
    ).float()
    sub = {
        "pointer": torch.arange(num_voxels + 1, device="cuda", dtype=torch.long),
        "value": torch.arange(num_voxels, device="cuda", dtype=torch.long),
    }

    with torch.no_grad():
        out = model(
            {
                "coord": coord,
                "feat": feat,
                "offset": offset,
                "segment": segment,
                "intensity": feat.clone(),
                "linearity": torch.zeros(num_voxels, 1, device="cuda"),
                "planarity": torch.zeros(num_voxels, 1, device="cuda"),
                "scattering": torch.zeros(num_voxels, 1, device="cuda"),
                "verticality": torch.zeros(num_voxels, 1, device="cuda"),
                "elevation": torch.zeros(num_voxels, 1, device="cuda"),
                "sub": sub,
                "num_raw_points": num_voxels,
                "grid_size": stage2["grid_size"],
            }
        )

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["seg_logits_superpoint"]).all()
    assert torch.isfinite(out["seg_logits_voxel"]).all()
    assert torch.isfinite(out["seg_logits"]).all()
    assert out["seg_logits"].shape[0] == num_voxels
