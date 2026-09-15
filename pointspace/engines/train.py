"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointspace.utils.comm as comm
from pointspace.datasets import build_dataset, point_collate_fn, collate_fn
from pointspace.datasets.sampler import (
    DistributedGuaranteedWeightedSampler,
    GuaranteedWeightedSampler,
)
from pointspace.models import build_model
from pointspace.utils.logger import get_root_logger
from pointspace.utils.optimizer import build_optimizer
from pointspace.utils.scheduler import build_scheduler
from pointspace.utils.events import EventStorage, ExceptionWriter
from pointspace.utils.registry import Registry
from .pcgrad import (
    accumulate_gradient_tuple,
    project_conflicting_gradient_group,
    subtract_gradient_tuples,
)


TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.epoch
        self.best_metric_value = -torch.inf
        self.best_metric_name = None
        self.best_metric_mode = "max"
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self._inject_class_weights()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self._gradient_accumulation_counter = 0

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Only clear gradients on first accumulation step
        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()

        # Forward pass
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = (
                output_dict["loss"] / self.cfg.gradient_accumulation_steps
            )  # scale loss

        # Inject per-criterion raw losses into output_dict for logging
        _model = self.model.module if hasattr(self.model, "module") else self.model
        _criteria = getattr(_model, "criteria", None)
        if _criteria is not None and hasattr(_criteria, "_last_individual_losses"):
            for name, raw in _criteria._last_individual_losses.items():
                output_dict[f"loss/{name}"] = raw

        # Backward pass
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._gradient_accumulation_counter += 1

        # Perform optimizer step only when enough gradients have accumulated
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            if self.cfg.enable_amp:
                self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.scaler.step(self.optimizer)

                # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
                # Fix torch warning scheduler step before optimizer step.
                scale = self.scaler.get_scale()
                self.scaler.update()
                if scale <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.optimizer.step()
                self.scheduler.step()

            # Reset grad accumulation counter
            self._gradient_accumulation_counter = 0

        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        # --- Sampler selection ---
        sample_weights = getattr(train_data, "sample_weights", None)
        use_weighted = (
            getattr(train_data, "weighted_sampler", False)
            and sample_weights is not None
        )

        loop = getattr(train_data, "loop", 1)
        num_base = len(train_data.data_list) if hasattr(train_data, "data_list") else len(train_data) // max(loop, 1)

        if comm.get_world_size() > 1:
            if use_weighted:
                train_sampler = DistributedGuaranteedWeightedSampler(
                    weights=sample_weights,
                    num_base_samples=num_base,
                    loop=loop,
                    num_replicas=comm.get_world_size(),
                    rank=comm.get_rank(),
                )
                self.logger.info(
                    f"Using DistributedGuaranteedWeightedSampler "
                    f"(base={num_base}, loop={loop}, total={num_base * loop})"
                )
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_data
                )
        else:
            if use_weighted:
                train_sampler = GuaranteedWeightedSampler(
                    weights=sample_weights,
                    num_base_samples=num_base,
                    loop=loop,
                )
                self.logger.info(
                    f"Using GuaranteedWeightedSampler "
                    f"(base={num_base}, loop={loop}, total={num_base * loop})"
                )
            else:
                train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        loader_workers = self.cfg.num_worker_per_gpu
        loader_kwargs = {}
        if loader_workers > 0:
            loader_kwargs["prefetch_factor"] = int(
                getattr(self.cfg, "prefetch_factor", 2)
            )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=loader_workers,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size_train,
            persistent_workers=(
                loader_workers > 0
                and bool(getattr(self.cfg, "persistent_workers", True))
            ),
            **loader_kwargs,
        )

        # Log batch-size breakdown
        grad_accum = self.cfg.gradient_accumulation_steps
        micro_bs = self.cfg.batch_size_per_gpu
        effective_bs = micro_bs * grad_accum
        self.logger.info(
            f"Train DataLoader: micro_batch={micro_bs}, "
            f"gradient_accumulation={grad_accum}, "
            f"effective_batch_per_gpu={effective_bs}"
        )
        if effective_bs != self.cfg.batch_size_train_per_gpu:
            self.logger.warning(
                f"batch_size_train_per_gpu ({self.cfg.batch_size_train_per_gpu}) "
                f"is not exactly divisible by gradient_accumulation_steps ({grad_accum}). "
                f"Actual effective batch = {effective_bs}."
            )
        return train_loader

    def _inject_class_weights(self):
        """Inject dataset-computed class weights into the model's criteria.

        Only affects loss functions with ``auto_class_weight=True``.
        Works for any model that exposes a ``criteria`` attribute (e.g.
        ``DefaultSegmentor``, ``DefaultSegmentorV2``), so the mechanism is
        not tied to a specific task.
        """
        def _requires_auto_class_weight(criteria_obj):
            if criteria_obj is None:
                return False
            inner = getattr(criteria_obj, "criteria", None)
            if inner is None:
                inner = [criteria_obj]
            return any(getattr(loss_obj, "auto_class_weight", False) for loss_obj in inner)

        train_data = self.train_loader.dataset
        class_weight = getattr(train_data, "class_weight", None)

        # Unwrap DDP to reach the real model
        model = self.model.module if hasattr(self.model, "module") else self.model
        criteria = getattr(model, "criteria", None)
        aux_criteria = getattr(model, "aux_criteria", None)

        need_class_weight = _requires_auto_class_weight(criteria) or _requires_auto_class_weight(
            aux_criteria
        )

        if class_weight is None:
            if need_class_weight:
                raise ValueError(
                    "Found loss with auto_class_weight=True, but the training dataset "
                    "did not provide class_weight. Set dataset.class_weight to a manual "
                    "list/array or a supported auto-compute mode."
                )
            return

        if hasattr(criteria, "set_class_weight"):
            criteria.set_class_weight(class_weight)
            self.logger.info(
                f"Injected dataset class_weight "
                f"(len={len(class_weight)}) into criteria"
            )

        # Also inject into aux_criteria if it exists (for DeepLASegmentor HDS)
        if aux_criteria is not None and hasattr(aux_criteria, "set_class_weight"):
            aux_criteria.set_class_weight(class_weight)
            self.logger.info(
                f"Injected dataset class_weight "
                f"(len={len(class_weight)}) into aux_criteria"
            )

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            loader_workers = self.cfg.num_worker_per_gpu
            loader_kwargs = {}
            if loader_workers > 0:
                loader_kwargs["prefetch_factor"] = int(
                    getattr(self.cfg, "val_prefetch_factor", 2)
                )
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=loader_workers,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
                **loader_kwargs,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader)
            * self.cfg.epoch
            // self.cfg.gradient_accumulation_steps
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("CVRLPCGradTrainer")
class CVRLPCGradTrainer(Trainer):
    """DefaultTrainer with exact optimizer-step CVRL gradient surgery.

    Every micro-batch contributes its weighted VPSD gradient to a private
    accumulator while ordinary backward accumulates the combined gradient.
    Only when ``gradient_accumulation_steps`` is reached is RRCL recovered by
    subtraction and the accumulated task gradients projected stage by stage.
    Task-private gradients, AMP unscaling and clipping remain unchanged.

    In distributed training, the VPSD gradient is averaged before subtraction,
    matching DDP's averaged combined gradient. All ranks therefore project the
    same global task gradients and remain synchronized.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        model = self.model.module if hasattr(self.model, "module") else self.model
        required = {
            "pcgrad_parameter_groups",
            "pcgrad_is_active",
            "pcgrad_should_log",
        }
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(
                "CVRLPCGradTrainer requires a PCGrad-aware CVRL model; "
                f"missing methods: {missing}"
            )
        groups = model.pcgrad_parameter_groups()
        self._pcgrad_group_names = tuple(groups)
        self._pcgrad_parameters = []
        self._pcgrad_slices = {}
        parameter_ids = set()
        for name, parameters in groups.items():
            start = len(self._pcgrad_parameters)
            for parameter in parameters:
                if id(parameter) in parameter_ids:
                    raise RuntimeError(
                        f"PCGrad parameter appears in more than one stage: {name}"
                    )
                parameter_ids.add(id(parameter))
                self._pcgrad_parameters.append(parameter)
            self._pcgrad_slices[name] = slice(
                start, len(self._pcgrad_parameters)
            )
        self._pcgrad_parameters = tuple(self._pcgrad_parameters)
        self._pcgrad_metric_sums = {}
        self._pcgrad_metric_count = 0
        self._pcgrad_window_active = False
        self._pcgrad_visual_accum = None
        self.logger.info(
            "CVRL PCGrad: symmetric optimizer-step stages=%s start_ratio=%.3f "
            "log_interval=%d shared_params=%d",
            ",".join(self._pcgrad_group_names),
            model.pcgrad_start_ratio,
            model.pcgrad_log_interval,
            sum(parameter.numel() for parameter in self._pcgrad_parameters),
        )

    def _visual_gradients(self, task_loss):
        accumulation = self.cfg.gradient_accumulation_steps
        objective = task_loss / accumulation
        if self.cfg.enable_amp:
            objective = self.scaler.scale(objective)
        return torch.autograd.grad(
            objective,
            self._pcgrad_parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

    def _project_task_gradients(self, visual_grad, return_grad):
        projected = [None] * len(self._pcgrad_parameters)
        metrics = {}
        for stage in self._pcgrad_group_names:
            group_slice = self._pcgrad_slices[stage]
            grad_v = visual_grad[group_slice]
            grad_r = return_grad[group_slice]
            model = self.model.module if hasattr(self.model, "module") else self.model
            projected_v, projected_r, stats = project_conflicting_gradient_group(
                grad_v,
                grad_r,
                eps=model.pcgrad_eps,
            )
            for local_index, (original_v, original_r, new_v, new_r) in enumerate(
                zip(grad_v, grad_r, projected_v, projected_r)
            ):
                if original_v is None and original_r is None:
                    continue
                combined = new_r if new_v is None else new_v
                if original_v is not None and original_r is not None:
                    combined = new_v + new_r
                projected[group_slice.start + local_index] = combined
            metrics.update(
                {f"{stage}_{key}": value for key, value in stats.items()}
            )

        stage_conflicts = torch.stack(
            [metrics[f"{stage}_conflict_rate"] for stage in self._pcgrad_group_names]
        )
        stage_projection = torch.stack(
            [metrics[f"{stage}_projection_ratio"] for stage in self._pcgrad_group_names]
        )
        metrics["conflict_rate"] = stage_conflicts.mean()
        metrics["projection_ratio"] = stage_projection.mean()
        return projected, metrics

    @staticmethod
    def _distributed_average_gradients(gradients, parameters):
        world_size = comm.get_world_size()
        if world_size <= 1:
            return gradients
        pieces = [
            (
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
            )
            for gradient, parameter in zip(gradients, parameters)
        ]
        flat = torch.cat(pieces)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)
        averaged = []
        begin = 0
        for parameter in parameters:
            length = parameter.numel()
            averaged.append(flat[begin : begin + length].view_as(parameter))
            begin += length
        return tuple(averaged)

    def _combined_accumulated_gradients(self):
        return tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in self._pcgrad_parameters
        )

    def _accumulate_pcgrad_metrics(self, metrics, output_dict, force_log=False):
        scale = self.scaler.get_scale() if self.cfg.enable_amp else 1.0
        # The task gradients have already been averaged over all micro-batches
        # by their loss/gradient_accumulation_steps factors.
        norm_rescale = 1.0 / float(scale)
        for key, value in metrics.items():
            value = value.detach().float()
            if key.endswith("_visual_norm") or key.endswith("_return_norm"):
                value = value * norm_rescale
            if key not in self._pcgrad_metric_sums:
                self._pcgrad_metric_sums[key] = value
            else:
                self._pcgrad_metric_sums[key] = (
                    self._pcgrad_metric_sums[key] + value
                )
        self._pcgrad_metric_count += 1

        model = self.model.module if hasattr(self.model, "module") else self.model
        if not (model.pcgrad_should_log() or force_log):
            return
        count = max(self._pcgrad_metric_count, 1)
        for key, value in self._pcgrad_metric_sums.items():
            output_dict[f"grad/{key}"] = value / count
        self._pcgrad_metric_sums.clear()
        self._pcgrad_metric_count = 0

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        model = self.model.module if hasattr(self.model, "module") else self.model
        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()
            self._pcgrad_window_active = model.pcgrad_is_active()
            self._pcgrad_visual_accum = None

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"] / self.cfg.gradient_accumulation_steps

        visual_loss = output_dict.pop("_pcgrad_visual_loss", None)
        return_loss = output_dict.pop("_pcgrad_return_loss", None)
        output_dict.pop("_pcgrad_log", None)
        # Activation is fixed at the beginning of an accumulation window so
        # the 10% boundary cannot create a partially projected optimizer step.
        pcgrad_active = self._pcgrad_window_active
        if pcgrad_active and (visual_loss is None or return_loss is None):
            raise RuntimeError(
                "Active CVRL PCGrad step did not expose both task losses"
            )

        criteria = getattr(model, "criteria", None)
        if criteria is not None and hasattr(criteria, "_last_individual_losses"):
            for name, raw in criteria._last_individual_losses.items():
                output_dict[f"loss/{name}"] = raw

        visual_grad = None
        if pcgrad_active:
            visual_grad = self._visual_gradients(visual_loss)
            visual_grad = self._distributed_average_gradients(
                visual_grad, self._pcgrad_parameters
            )
            self._pcgrad_visual_accum = accumulate_gradient_tuple(
                self._pcgrad_visual_accum, visual_grad
            )

        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._gradient_accumulation_counter += 1
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            if self._pcgrad_window_active:
                if self._pcgrad_visual_accum is None:
                    raise RuntimeError("Active PCGrad window has no Visual gradients")
                combined_grad = self._combined_accumulated_gradients()
                return_grad = subtract_gradient_tuples(
                    combined_grad, self._pcgrad_visual_accum
                )
                projected_grad, metrics = self._project_task_gradients(
                    self._pcgrad_visual_accum, return_grad
                )
                with torch.no_grad():
                    for parameter, projected in zip(
                        self._pcgrad_parameters, projected_grad
                    ):
                        if projected is None:
                            continue
                        if parameter.grad is None:
                            raise RuntimeError(
                                "PCGrad shared parameter has no accumulated gradient"
                            )
                        parameter.grad.copy_(projected.to(parameter.grad.dtype))
                final_step = (
                    self.epoch + 1 == self.max_epoch
                    and self.comm_info["iter"] + 1 == len(self.train_loader)
                )
                self._accumulate_pcgrad_metrics(
                    metrics, output_dict, force_log=final_step
                )
            if self.cfg.enable_amp:
                self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.scaler.step(self.optimizer)
                scale = self.scaler.get_scale()
                self.scaler.update()
                if scale <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.optimizer.step()
                self.scheduler.step()
            self._gradient_accumulation_counter = 0
            self._pcgrad_window_active = False
            self._pcgrad_visual_accum = None

        self.comm_info["model_output_dict"] = output_dict


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointspace.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
