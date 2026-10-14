"""Generate standalone downstream configs for the Section 4.5 SSL matrix.

The generated files deliberately do not use ``_base_`` inheritance.  Edit the
tables below and rerun this file when the controlled protocol changes.
"""

import argparse
from pathlib import Path
from pprint import pformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "configs" / "downstream"

METHODS = (
    "scratch",
    "point-m2ae",
    "msc",
    "sonata",
    "d-ditr",
    "concerto",
    "vpsd",
    "rrcl",
    "cvrl",
)
PROBES = ("lin", "dec", "ft")

# Uniform downstream protocol selected from dense-sample full-fine-tuning
# measurements on an RTX 4080 (16 GiB).  DALES-CVRL is the limiting case at
# about 14.20 GiB reserved memory for micro batch 6.
MICRO_BATCH_SIZE = 6
GRADIENT_ACCUMULATION_STEPS = 2

CHECKPOINTS = {
    # Frozen scratch probes need meaningful normalization statistics too.  The
    # learnable tensors are still the exact seed-42 random initialization;
    # only BN buffers are estimated on the same clean Swiss source set used
    # for every pretrained checkpoint.
    "scratch": r"exp\rrcl\swiss\scratch-litept-v1m4-100\model\random_seed42_bncal_swiss.pth",
    "point-m2ae": r"exp\baselines\swiss\pretrain-point-m2ae-litept-v1m4-100\model\backbone_last_bncal_swiss.pth",
    "msc": r"exp\baselines\swiss\pretrain-msc-litept-v1m4-100\model\backbone_last_bncal_swiss.pth",
    "sonata": r"exp\baselines\swiss\pretrain-sonata-litept-v1m4-100\model\backbone_last_bncal_swiss.pth",
    "d-ditr": r"exp\baselines\swiss\pretrain-d-ditr-litept-v1m4-100\model\backbone_last_bncal_swiss.pth",
    "concerto": r"exp\baselines\swiss\pretrain-concerto-litept-v1m4-100\model\backbone_last_bncal_swiss.pth",
    "vpsd": r"exp\vpsd\swiss\pretrain-v1m1-litept-v1m4-100\model\model_last_bncal_swiss.pth",
    "rrcl": r"exp\rrcl\swiss\pretrain-v1m1-litept-v1m4-100\model\backbone_last_bncal_swiss.pth",
    "cvrl": r"exp\cvrl\swiss\pretrain-v1m1-litept-v1m4-100-full\model\model_last_bncal_swiss.pth",
}

DATASETS = {
    "dales": {
        "train": r"E:\data\DALES\dales_las\tile\train",
        "val": r"E:\data\DALES\dales_las\tile\test",
        "test": r"E:\data\DALES\dales_las\tile\test",
        "classes": list(range(1, 9)),
        "names": [
            "ground", "vegetation", "cars", "trucks", "power lines",
            "fences", "poles", "buildings",
        ],
        "grid_size": 0.25,
        "zero_intensity": True,
        # P99.5=172k and max=199k after 0.25 m voxel sampling.  The 200k cap
        # preserves essentially every native tile while bounding augmented
        # high-density outliers.
        "point_max": 200000,
        # A representative CVRL-FT batch used 8.35 GiB at micro 12 and
        # 10.75 GiB at micro 16.  Twelve leaves room for dense-tile mixtures.
        "micro_batch_size": 12,
        "batch_size_val": 8,
        "batch_size_test": 8,
    },
    "vaihingen": {
        "early_stopping": True,
        "train": r"E:\data\Vaihingen\tile\train",
        "val": r"E:\data\Vaihingen\tile\test",
        "test": r"E:\data\Vaihingen\tile\test",
        # ISPRS labels are already zero based in these LAS files.  Class 0 is
        # Powerline (546 official training points), not a noise class.
        "classes": list(range(9)),
        "names": [
            "powerline", "low vegetation", "impervious surfaces", "car",
            "fence/hedge", "roof", "facade", "shrub", "tree",
        ],
        "grid_size": 0.25,
        "zero_intensity": False,
        # All 58 training tiles are below 74k after voxel sampling.  This is an
        # inactive safety guard with headroom for RandomScale.
        "point_max": 90000,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "batch_size_val": 8,
        "batch_size_test": 8,
    },
    "opengf": {
        "epoch": 20,
        "train_loop": 3,
        "early_stopping": True,
        "train": r"E:\data\OpenGF\tile\train",
        # Use the official held-out test tiles for both checkpoint selection
        # and precise testing, matching the DALES/Vaihingen protocol.
        "val": r"E:\data\OpenGF\tile\test",
        "test": r"E:\data\OpenGF\tile\test",
        # OpenGF class 0 is noise/unclassified and is filtered before all
        # augmentations and voxel sampling.
        "classes": [1, 2],
        "names": ["non-ground", "ground"],
        "grid_size": 0.5,
        "zero_intensity": False,
        # P99.5=203k and max=262k after 0.5 m voxel sampling.  Keep nearly all
        # tiles while bounding the few extreme-density blocks.
        "point_max": 240000,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "batch_size_val": 2,
        "batch_size_test": 2,
    },
}


def litept_backbone(*, freeze_encoder=False):
    return {
        "type": "LitePT-v1m4",
        "in_channels": 4,
        "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        "stride": (2, 2, 2, 2),
        "enc_depths": (2, 2, 2, 6, 2),
        "enc_channels": (36, 72, 144, 252, 504),
        "enc_num_head": (2, 4, 8, 14, 28),
        "enc_patch_size": (192, 192, 192, 192, 192),
        "enc_conv": (True, True, True, False, False),
        "enc_attn": (False, False, False, True, True),
        "enc_rope_freq": (100.0,) * 5,
        "dec_depths": (1, 1, 1, 1),
        "dec_channels": (72, 72, 144, 252),
        "dec_num_head": (4, 4, 8, 14),
        "dec_patch_size": (192, 192, 192, 192),
        "dec_conv": (True, True, True, True),
        "dec_attn": (False, False, False, False),
        "dec_rope_freq": (100.0,) * 4,
        "dec_rope_enable": True,
        # Match the original stable CVRL downstream protocol.  In particular,
        # never use stateless BN here: test-time fragments are collated in
        # changing groups, so current-batch statistics would make a point's
        # prediction depend on the fragment beside it.
        "dec_norm": "batch",
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        # A timm-style DropPath sees flattened [N_point, C] LitePT tensors as
        # N independent samples.  That is point-wise residual corruption, not
        # image-style sample-wise stochastic depth, and caused a severe
        # train/eval mismatch in dense transfer.  Downstream transfer is
        # therefore deterministic for every method and every probe mode.
        "drop_path": 0.0,
        "pre_norm": True,
        "shuffle_orders": True,
        "enable_flash": True,
        "traceable": True,
        "mask_token": False,
        "enc_mode": False,
        "freeze_encoder": freeze_encoder,
        "shift_coords": None,
        "jitter_coords": None,
        "rescale_coords": None,
        "checkpoint_mlp": True,
    }


def cvrl_backbone():
    return {
        "type": "CVRL-v1m1-backbone",
        "mode": "full",
        "level_channels": (36, 72, 144, 252, 504),
        "distill_level": 2,
        "visual_channels": 128,
        "return_channels": 72,
        "native_channels": 1024,
        "native_hidden_channels": 1024,
        "pca_path": None,
        "require_pca": False,
        "pca_output_normalize": True,
        "pca_output_scale": 128 ** 0.5,
        "visual_token_chunk_size": 4096,
        "return_backbone": litept_backbone(freeze_encoder=False),
    }


def vpsd_backbone():
    encoder = litept_backbone(freeze_encoder=False)
    return {
        "type": "VPSD-v1m1-backbone",
        "feature_source": "visual",
        "normalize": False,
        "distill_level": 2,
        "level_channels": (36, 72, 144, 252, 504),
        "teacher_channels": 1024,
        "visual_channels": 128,
        "pca_path": None,
        "require_pca": False,
        "pca_output_normalize": True,
        "pca_output_scale": 128 ** 0.5,
        "edge_weight": "sqrt_count",
        "sample_balanced": True,
        "validate_mapping": False,
        "fuse_deeper_features": True,
        "projector_hidden_channels": 1024,
        "projector_checkpoint": True,
        "aggregate_channel_chunk_size": 64,
        "projector_patch_chunk_size": 4096,
        "normalize_dtype": "amp",
        "backbone": encoder,
    }


def criteria(num_classes):
    return [
        {
            "type": "CrossEntropyLoss",
            "loss_weight": 1.0,
            "ignore_index": -1,
            "auto_class_weight": True,
        },
        {
            "type": "LovaszLoss",
            "mode": "multiclass",
            "loss_weight": 1.0,
            "ignore_index": -1,
        },
    ]


def model_config(method, probe, num_classes):
    loss = criteria(num_classes)
    if method == "vpsd" and probe == "lin":
        return {
            "type": "FrozenBackboneLinearProbeSegmentor",
            "num_classes": num_classes,
            "backbone_out_channels": 128,
            "backbone": vpsd_backbone(),
            "criteria": loss,
        }
    if method == "cvrl" and probe == "lin":
        return {
            "type": "CVRLLinearProbeSegmentor",
            "num_classes": num_classes,
            "feature_source": "full",
            "backbone_out_channels": 200,
            "backbone": cvrl_backbone(),
            "criteria": loss,
        }
    if method == "cvrl" and probe == "ft":
        return {
            "type": "DefaultSegmentorV2",
            "num_classes": num_classes,
            "backbone_out_channels": 200,
            "backbone": cvrl_backbone(),
            "criteria": loss,
            "freeze_backbone": False,
        }
    backbone = litept_backbone(freeze_encoder=(probe == "dec"))
    return {
        "type": (
            "LitePTLinearProbeSegmentor"
            if probe == "lin"
            else "DefaultSegmentorV2"
        ),
        "num_classes": num_classes,
        "backbone_out_channels": 72,
        "backbone": backbone,
        "criteria": loss,
        **({} if probe == "lin" else {"freeze_backbone": False}),
    }


def checkpoint_hook(method, probe):
    if method == "scratch" and probe == "ft":
        return None
    if method == "vpsd" and probe == "lin":
        return {
            "type": "CheckpointLoader",
            "keywords": "module.",
            "replacement": "module.backbone.",
            "include_prefixes": (
                "backbone.", "student_projector.", "visual_projection."
            ),
            "strict": False,
            "required": True,
        }
    if method == "cvrl" and probe == "dec":
        return {
            "type": "CheckpointLoader",
            "keywords": "backbone.return_backbone.",
            "replacement": "backbone.",
            "include_prefixes": (
                "backbone.return_backbone.embedding.",
                "backbone.return_backbone.enc.",
            ),
            "strict": False,
            "required": True,
        }
    if probe == "dec":
        prefixes = ("backbone.embedding.", "backbone.enc.")
    else:
        prefixes = ("backbone.",)
    return {
        "type": "CheckpointLoader",
        "include_prefixes": prefixes,
        "strict": False,
        "required": True,
    }


def dataset_config(spec, *, label_fraction=None):
    intensity_transform = (
        {"type": "EnsureFeature", "key": "intensity", "channels": 1,
         "value": 0.0, "overwrite": True}
        if spec["zero_intensity"]
        else {"type": "RobustLogIntensity"}
    )
    common = {
        "type": "LasDataset",
        "required_class": spec["classes"],
        "remap_class": True,
        "ignore_index": -1,
    }
    train_transforms = [
        # required_class remapping has already mapped excluded/noise labels to
        # -1. Remove them before augmentation so they cannot affect features.
        {"type": "ClassFilter",
         "keep_classes": list(range(len(spec["classes"])))},
        {"type": "ZPercentileCenterShift", "percentile": 2.0},
        intensity_transform,
        {"type": "RandomDropout", "dropout_ratio": 0.2,
         "dropout_application_ratio": 0.2},
        {"type": "RandomRotate", "angle": [-1, 1], "axis": "z",
         "center": [0, 0, 0], "p": 0.5},
        {"type": "RandomScale", "scale": [0.9, 1.1]},
        {"type": "RandomFlip", "p": 0.5},
        {"type": "RandomJitter", "sigma": 0.005, "clip": 0.02},
        {"type": "GridSample", "grid_size": spec["grid_size"],
         "hash_type": "fnv", "mode": "train",
         "return_grid_coord": True},
    ]
    if label_fraction is not None:
        train_transforms.insert(1, {"type": "SparseLabelMask", "ignore_index": -1})
    if spec.get("point_max") is not None:
        train_transforms.append(
            {"type": "SphereCrop", "point_max": spec["point_max"],
             "mode": "random"}
        )

    train = {
        **common,
        "split": "train",
        "data_path": spec["train"],
        "class_weight": "sqrt",
        "weight_sample": 0.2,
        "weighted_sampler": True,
        "test_mode": False,
        "loop": spec.get("train_loop", 5),
        "transform": train_transforms,
        "post_transform": [
            {"type": "ToTensor"},
            {"type": "Collect", "keys": ("coord", "segment", "grid_coord"),
             "feat_keys": ("coord", "intensity")},
        ],
    }
    if label_fraction is not None:
        train.update({
            "type": "SparseLabelLasDataset",
            "label_manifest_dir": r"E:\data\DALES\dales_las\tile\sparse_labels_seed42",
            "label_fraction": label_fraction,
        })
    val = {
        **common,
        "split": "val",
        "data_path": spec["val"],
        "test_mode": False,
        "loop": 1,
        "transform": [
            {"type": "Copy", "keys_dict": {"segment": "origin_segment"}},
            {"type": "ZPercentileCenterShift", "percentile": 2.0},
            intensity_transform,
            {"type": "GridSample", "grid_size": spec["grid_size"],
             "hash_type": "fnv", "mode": "train",
             "return_grid_coord": True, "return_inverse": True},
        ],
        "post_transform": [
            {"type": "ToTensor"},
            {"type": "Collect",
             "keys": ("coord", "grid_coord", "segment", "origin_segment", "inverse"),
             "feat_keys": ("coord", "intensity")},
        ],
    }
    test = {
        **common,
        "split": "test",
        "data_path": spec["test"],
        "test_mode": True,
        "transform": [
            {"type": "ZPercentileCenterShift", "percentile": 2.0},
            intensity_transform,
            {"type": "GridSample", "grid_size": spec["grid_size"],
             "hash_type": "fnv", "mode": "test",
             "return_grid_coord": True, "return_inverse": True},
        ],
        "aug_transform": [[{"type": "RandomScale", "scale": [1.0, 1.0]}]],
        "post_transform": [
            {"type": "ToTensor"},
            {"type": "Collect", "keys": ("coord", "grid_coord", "index"),
             "feat_keys": ("coord", "intensity")},
        ],
    }
    return {
        "num_classes": len(spec["classes"]),
        "ignore_index": -1,
        "names": spec["names"],
        "train": train,
        "val": val,
        "test": test,
    }


def precise_bn_data(spec):
    """Clean downstream-train distribution used to estimate BN buffers."""
    intensity_transform = (
        {"type": "EnsureFeature", "key": "intensity", "channels": 1,
         "value": 0.0, "overwrite": True}
        if spec["zero_intensity"]
        else {"type": "RobustLogIntensity"}
    )
    transforms = [
        {"type": "ClassFilter",
         "keep_classes": list(range(len(spec["classes"])))},
        {"type": "ZPercentileCenterShift", "percentile": 2.0},
        intensity_transform,
        {"type": "GridSample", "grid_size": spec["grid_size"],
         "hash_type": "fnv", "mode": "train",
         "return_grid_coord": True},
    ]
    if spec.get("point_max") is not None:
        transforms.append(
            {"type": "SphereCrop", "point_max": spec["point_max"],
             "mode": "center"}
        )
    return {
        "type": "LasDataset",
        "required_class": spec["classes"],
        "remap_class": True,
        "ignore_index": -1,
        "split": "train",
        "data_path": spec["train"],
        "test_mode": False,
        "loop": 1,
        "transform": transforms,
        "post_transform": [
            {"type": "ToTensor"},
            {"type": "Collect", "keys": ("coord", "grid_coord"),
             "feat_keys": ("coord", "intensity")},
        ],
    }


def render(
    dataset_name,
    method,
    probe,
    *,
    label_fraction=None,
    save_path_root=None,
    experiment_seed=42,
    label_efficiency_full=False,
):
    if label_fraction is not None and (dataset_name != "dales" or probe != "ft"):
        raise ValueError("Sparse-label experiments are DALES full fine-tuning only")
    spec = DATASETS[dataset_name]
    if label_efficiency_full and (dataset_name != "dales" or probe != "ft"):
        raise ValueError("Full-label efficiency controls are DALES full fine-tuning only")
    if label_fraction is not None or label_efficiency_full:
        # Section 4.6 schedule. Keep the completed 4.5 DALES configs intact.
        spec = {**spec, "epoch": 20, "train_loop": 3, "early_stopping": True}
    weight = None if method == "scratch" and probe == "ft" else CHECKPOINTS[method]
    model = model_config(method, probe, len(spec["classes"]))
    loader = checkpoint_hook(method, probe)
    hooks = [] if loader is None else [loader]
    hooks.extend([
        {"type": "RuntimeInfoHook"},
        {"type": "ModelHook"},
        {"type": "IterationTimer", "warmup_iter": 2},
        {"type": "InformationWriter", "interval": 10},
        {"type": "CacheCleaner", "time_multiplier": 5,
         "step_clean_interval": 200},
    ])
    # LP preserves the source-calibrated frozen representation exactly.  A
    # decoder probe re-estimates only its new decoder BN; FT lets all running
    # statistics adapt during optimization and then estimates every BN buffer
    # once more on clean downstream training data immediately before val.
    if probe in {"dec", "ft"}:
        hooks.append({
            "type": "PreciseBatchNormHook",
            "data": precise_bn_data(spec),
            "scope": "decoder" if probe == "dec" else "all",
            "num_batches": 64,
            "batch_size": spec.get("batch_size_val", 2),
            "num_workers": 2,
            "seed": experiment_seed,
            "interval": 1,
            "enabled": True,
        })
    hooks.extend([
        {"type": "SemSegEvaluator", "write_cls_iou": True,
         "log_interval": 10},
        {"type": "CheckpointSaver", "save_freq": None},
    ])
    if spec.get("early_stopping", False):
        hooks.append({
            "type": "EarlyStoppingHook",
            "monitor": "mIoU",
            "mode": "max",
            "source": "comm_info",
            "patience": 3,
            "min_delta_percent": 0.25,
            "warmup_epochs": 2,
        })
    hooks.append({"type": "PreciseEvaluator", "test_last": False})
    base_lr = 2e-3 if probe in {"lin", "dec"} else 1e-3
    # Full fine-tuning starts immediately and uses one LR for the complete
    # network.  No encoder LR reduction and no decoder-only warm-up.
    param_dicts = []
    values = {
        "dataset_name": dataset_name,
        "pretrain_method": method,
        "probe_mode": probe,
        "train_data_dir": spec["train"],
        "val_data_dir": spec["val"],
        "test_data_dir": spec["test"],
        "save_path": (
            f"{save_path_root or 'exp/label_efficiency/dales'}/{label_fraction:g}/{method}-ft"
            if label_fraction is not None
            else (
                f"{save_path_root}/1/{method}-ft"
                if label_efficiency_full and save_path_root is not None
                else f"exp/downstream/{dataset_name}/{method}-{probe}"
            )
        ),
        "weight": weight,
        "resume": False,
        "evaluate": True,
        "test_only": False,
        "seed": experiment_seed,
        "batch_size_train": (
            spec.get("micro_batch_size", MICRO_BATCH_SIZE)
            * GRADIENT_ACCUMULATION_STEPS
        ),
        "batch_size_val": spec.get("batch_size_val", 2),
        "batch_size_test": spec.get("batch_size_test", 2),
        "num_worker": 2,
        "prefetch_factor": 1,
        "val_prefetch_factor": 1,
        "persistent_workers": False,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epoch": spec.get("epoch", 10),
        "clip_grad": 3.0 if probe == "ft" else None,
        "enable_amp": True,
        "amp_dtype": "bfloat16",
        "sync_bn": False,
        "find_unused_parameters": False,
        "mix_prob": 0.0,
        "enable_wandb": False,
        "wandb_project": "pointspace-cvrl-downstream",
        "wandb_key": None,
        "model": model,
        "optimizer": {"type": "AdamW", "lr": base_lr,
                      "weight_decay": 2e-3},
        "scheduler": {"type": "CosineAnnealingLR", "total_steps": spec.get("epoch", 10)},
        "param_dicts": param_dicts,
        "hooks": hooks,
        "train": {"type": "DefaultTrainer"},
        "test": {"type": "SemSegTester"},
        # No writer: precise testing logs metrics without creating LAS files.
        "writer": None,
        "data": dataset_config(spec, label_fraction=label_fraction),
    }
    header = (
        "# Auto-generated by utils/generate_downstream_ssl_configs.py.\n"
        "# Standalone config: no _base_ dependency.\n"
        f"# Dataset={dataset_name}; pretraining={method}; probe={probe}; "
        f"label_fraction={label_fraction}; experiment_seed={experiment_seed}; "
        f"label_efficiency_full={label_efficiency_full}.\n\n"
    )
    return header + "\n\n".join(
        f"{key} = {pformat(value, width=100, sort_dicts=False)}"
        for key, value in values.items()
    ) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate standalone Section 4.5 downstream configs."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets to regenerate; omit to regenerate the complete matrix.",
    )
    args = parser.parse_args()
    selected_datasets = tuple(args.datasets) or tuple(DATASETS)
    invalid = sorted(set(selected_datasets).difference(DATASETS))
    if invalid:
        parser.error(
            f"unknown dataset(s): {', '.join(invalid)}; "
            f"choose from {', '.join(DATASETS)}"
        )
    count = 0
    for dataset_name in selected_datasets:
        output_dir = OUTPUT_ROOT / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            for probe in PROBES:
                path = output_dir / f"semseg-{method}-litept-v1m4-{dataset_name}-{probe}.py"
                path.write_text(render(dataset_name, method, probe), encoding="utf-8")
                count += 1
    print(f"Generated {count} standalone configs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
