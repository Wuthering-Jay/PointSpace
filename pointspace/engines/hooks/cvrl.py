"""Training schedules and fixed representation audits for CVRL."""

import json
import os
from pathlib import Path
from uuid import uuid4

import torch
from safetensors.torch import save_file as save_safetensors

import pointspace.utils.comm as comm

from .builder import HOOKS
from .default import HookBase


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


@HOOKS.register_module()
class CVRLTrainingSchedule(HookBase):
    """Send global training progress to scheduled CVRL variants."""

    def before_train(self):
        model = _unwrap_model(self.trainer.model)
        if not hasattr(model, "set_training_progress"):
            raise TypeError(
                "CVRLTrainingSchedule requires a model with "
                "set_training_progress(progress, step)"
            )

    def before_step(self):
        steps_per_epoch = len(self.trainer.train_loader)
        total_steps = max(self.trainer.max_epoch * steps_per_epoch, 1)
        step = self.trainer.epoch * steps_per_epoch + self.trainer.comm_info["iter"]
        progress = step / max(total_steps - 1, 1)
        _unwrap_model(self.trainer.model).set_training_progress(progress, step)


@HOOKS.register_module()
class CVRLAuditFeatureSaver(HookBase):
    """Save the same small validation subset and all CVRL branches each epoch.

    The validation loader must be deterministic and retain ``segment``. Native
    and PCA teachers are stored sparsely through ``teacher_point_index`` so
    image-invisible points do not allocate fake 1024D rows.
    """

    def __init__(
        self,
        output_dir=None,
        num_batches=8,
        max_points=32768,
        feature_dtype="float16",
    ):
        self.output_dir = output_dir
        self.num_batches = int(num_batches)
        self.max_points = int(max_points)
        self.feature_dtype = str(feature_dtype)
        if self.num_batches <= 0 or self.max_points <= 0:
            raise ValueError("audit num_batches and max_points must be positive")
        if self.feature_dtype not in {"float16", "float32"}:
            raise ValueError("audit feature_dtype must be float16 or float32")

    def _output_path(self):
        root = Path(
            self.output_dir
            if self.output_dir is not None
            else Path(self.trainer.cfg.save_path) / "audit_features"
        )
        root.mkdir(parents=True, exist_ok=True)
        return root / f"epoch_{self.trainer.epoch + 1:04d}.safetensors"

    @staticmethod
    def _batch_ids(offset, length, base):
        batch = torch.empty(length, dtype=torch.int64, device=offset.device)
        begin = 0
        for index, end in enumerate(offset.long().tolist()):
            batch[begin:end] = base + index
            begin = end
        return batch

    def after_epoch(self):
        if not comm.is_main_process():
            return
        if self.trainer.val_loader is None:
            if self.trainer.epoch == 0:
                self.trainer.logger.info(
                    "CVRL audit disabled because evaluate=False and no "
                    "validation loader was built"
                )
            return
        model = _unwrap_model(self.trainer.model)
        if not hasattr(model, "backbone"):
            raise TypeError("CVRLAuditFeatureSaver requires a CVRL wrapper model")
        backbone = model.backbone
        active_mode = str(getattr(model, "mode", "full")).lower()
        was_training = self.trainer.model.training
        self.trainer.model.eval()
        dtype = torch.float16 if self.feature_dtype == "float16" else torch.float32
        collected = {
            "coord": [],
            "label": [],
            "sample_id": [],
        }
        if active_mode in {"visual", "full"}:
            collected["z_visual"] = []
            collected["z_visual_native"] = []
        if active_mode in {"return", "full"}:
            collected["z_return"] = []
        if active_mode == "full":
            collected["z_full"] = []
        teacher_native = []
        teacher_pca = []
        teacher_point_index = []
        point_base = 0

        amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.trainer.cfg.amp_dtype]
        with torch.no_grad():
            for batch_index, input_dict in enumerate(self.trainer.val_loader):
                if batch_index >= self.num_batches or point_base >= self.max_points:
                    break
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.cuda(non_blocking=True)
                required = {"coord", "offset", "segment"}
                if active_mode in {"visual", "full"}:
                    required.update(
                        {"dino_feature", "image_patch_index", "image_valid"}
                    )
                missing = required.difference(input_dict)
                if missing:
                    raise KeyError(
                        "CVRL audit validation input is missing: "
                        f"{sorted(missing)}"
                    )
                with torch.autocast(
                    device_type="cuda",
                    enabled=self.trainer.cfg.enable_amp,
                    dtype=amp_dtype,
                ):
                    point, branches = backbone(
                        input_dict, return_branches=True, mode=active_mode
                    )

                remaining = self.max_points - point_base
                remaining_batches = self.num_batches - batch_index
                # Reserve a quota for later deterministic batches instead of
                # exhausting the audit budget on the first few tiles.
                quota = max(1, remaining // max(remaining_batches, 1))
                count = min(int(point.feat.shape[0]), quota, remaining)
                if count <= 0:
                    break
                if count == point.feat.shape[0]:
                    selected = torch.arange(count, device=point.feat.device)
                else:
                    selected = torch.linspace(
                        0,
                        point.feat.shape[0] - 1,
                        count,
                        device=point.feat.device,
                    ).round().long()
                sample_id = self._batch_ids(
                    input_dict["offset"],
                    point.feat.shape[0],
                    batch_index * 100000,
                )[selected]
                collected["coord"].append(input_dict["coord"][selected].float().cpu())
                collected["label"].append(input_dict["segment"][selected].long().cpu())
                collected["sample_id"].append(sample_id.cpu())
                visual = return_feature = None
                if active_mode in {"visual", "full"}:
                    visual = branches["visual_feature"][selected]
                    collected["z_visual"].append(visual.to(dtype).cpu())
                    # Project only the P2 tokens referenced by the fixed audit
                    # rows. This exposes the 1024D pre-PCA representation
                    # without materializing native features for every point.
                    compact = branches["compact"]
                    fused = backbone.fuse_token_hierarchy(
                        compact, branches["level_features"]
                    )
                    token_index, token_inverse = torch.unique(
                        compact.input_to_target[selected],
                        sorted=True,
                        return_inverse=True,
                    )
                    native = backbone.semantic_projector(fused[token_index])[
                        token_inverse
                    ]
                    collected["z_visual_native"].append(
                        native.to(dtype).cpu()
                    )
                if active_mode in {"return", "full"}:
                    return_feature = branches["return_feature"][selected]
                    collected["z_return"].append(return_feature.to(dtype).cpu())
                if active_mode == "full":
                    full = torch.cat((visual, return_feature), dim=-1)
                    collected["z_full"].append(full.to(dtype).cpu())

                if active_mode in {"visual", "full"}:
                    patch_index = input_dict["image_patch_index"][selected].long()
                    teacher_valid = input_dict["image_valid"][selected].bool()
                    teacher_valid &= patch_index >= 0
                    teacher_valid &= patch_index < input_dict["dino_feature"].shape[0]
                    if teacher_valid.any():
                        raw_teacher = input_dict["dino_feature"][
                            patch_index[teacher_valid]
                        ]
                        reduced_teacher = backbone.visual_projection(raw_teacher)
                        teacher_native.append(raw_teacher.to(dtype).cpu())
                        teacher_pca.append(reduced_teacher.to(dtype).cpu())
                        teacher_point_index.append(
                            torch.nonzero(teacher_valid, as_tuple=False).flatten().cpu()
                            + point_base
                        )
                point_base += count

        if not collected["coord"]:
            raise RuntimeError("CVRL audit loader produced no points")
        tensors = {
            key: torch.cat(value).contiguous()
            for key, value in collected.items()
        }
        if teacher_native:
            tensors.update(
                teacher_native=torch.cat(teacher_native).contiguous(),
                teacher_pca=torch.cat(teacher_pca).contiguous(),
                teacher_point_index=torch.cat(teacher_point_index).contiguous(),
            )
        else:
            tensors.update(
                teacher_native=torch.empty((0, backbone.native_channels), dtype=dtype),
                teacher_pca=torch.empty((0, backbone.visual_channels), dtype=dtype),
                teacher_point_index=torch.empty(0, dtype=torch.int64),
            )

        output_path = self._output_path()
        temporary_path = output_path.with_name(
            f".{output_path.name}.{uuid4().hex}.tmp"
        )
        metadata = {
            "format_version": "1",
            "architecture": str(self.trainer.cfg.model.type),
            "epoch": str(self.trainer.epoch + 1),
            "feature_dtype": self.feature_dtype,
            "feature_layout": (
                "z_visual_native->PCA=z_visual;z_visual|z_return=z_full"
                if active_mode == "full"
                else "z_visual_native->PCA=z_visual"
                if active_mode == "visual"
                else f"z_{active_mode}"
            ),
            "fixed_subset": "deterministic_validation_prefix",
            "num_batches": str(self.num_batches),
            "max_points": str(self.max_points),
            "teacher_storage": "sparse_by_teacher_point_index",
            "focus_class_defaults": json.dumps(
                {"building": 6, "road": 11, "bridge": 17}
            ),
        }
        try:
            save_safetensors(tensors, str(temporary_path), metadata)
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self.trainer.logger.info(
            "CVRL audit saved: %s, points=%d, teacher-visible=%d",
            output_path,
            tensors["coord"].shape[0],
            tensors["teacher_point_index"].shape[0],
        )
        if was_training:
            self.trainer.model.train()
