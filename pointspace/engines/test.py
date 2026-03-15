"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import json
from uuid import uuid4
import os
import time
import numpy as np
from collections import OrderedDict
from functools import partial
from packaging import version
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_closing, label as _ndimage_label
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from scipy.spatial import ConvexHull
from matplotlib.path import Path

from .defaults import create_ddp_model
import pointspace.utils.comm as comm
from pointspace.datasets import build_dataset, collate_fn
from pointspace.models import build_model
from pointspace.writers import build_writer
from pointspace.writers.benchmark import create_benchmark_writer
from pointspace.utils.logger import get_root_logger
from pointspace.utils.registry import Registry
from pointspace.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

try:
    import pointops
except:
    pointops = None


TESTERS = Registry("testers")

AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


def _build_autocast(cfg):
    """Return a context-manager factory matching the AMP setup used in training.

    Usage inside eval / test loops::

        auto_cast = _build_autocast(self.cfg)
        with torch.no_grad(), auto_cast():
            output = self.model(input_dict)
    """
    enable = getattr(cfg, "enable_amp", False)
    dtype_key = getattr(cfg, "amp_dtype", "float16")
    dtype = AMP_DTYPE.get(dtype_key, torch.float16)
    if version.parse(torch.__version__) >= version.parse("2.4"):
        return partial(torch.amp.autocast, device_type="cuda", enabled=enable, dtype=dtype)
    else:
        return partial(torch.cuda.amp.autocast, enabled=enable, dtype=dtype)


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=True) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        # Build CacheCleaner for test phase (reuse training hook config if available)
        self.cache_cleaner = self._build_cache_cleaner()
        if self.verbose and model is None:
            # if model is not none, trigger tester with trainer, no need to print config
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
            # Restore class mapping saved at training time so the writer can
            # inverse-map predicted remapped IDs back to original class IDs.
            id2class = checkpoint.get("id2class", None)
            self.id2class = id2class  # dict {remapped_id: original_id} or None
            if id2class is not None:
                self.logger.info(
                    f"=> Loaded id2class mapping from checkpoint: {id2class}"
                )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def _remap_to_original(self, pred: np.ndarray) -> np.ndarray:
        """Apply id2class to convert remapped IDs back to original class IDs.

        Used only for file output (LAS writer etc.); metrics are always
        computed in the remapped (contiguous) ID space.

        Args:
            pred: integer numpy array of remapped class IDs.

        Returns:
            Array with original class IDs, or the input unchanged if no
            mapping is available.
        """
        id2class = getattr(self, "id2class", None)
        if id2class is None:
            return pred
        out = np.empty_like(pred)
        for remapped, original in id2class.items():
            out[pred == remapped] = original
        return out

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    def _build_cache_cleaner(self):
        """Build a CacheCleaner for the test phase.

        Looks for a CacheCleaner config in ``cfg.hooks`` (the same list used
        during training) and reuses its parameters.  Falls back to a default
        CacheCleaner if none is configured.
        """
        from pointspace.engines.hooks.misc import CacheCleaner

        hooks_cfg = getattr(self.cfg, "hooks", None) or []
        cc_cfg = {}
        for h in hooks_cfg:
            if isinstance(h, dict) and h.get("type") == "CacheCleaner":
                cc_cfg = {k: v for k, v in h.items() if k != "type"}
                break
        cleaner = CacheCleaner(**cc_cfg)
        cleaner.logger = self.logger
        return cleaner

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        batch_size_test = self.cfg.batch_size_test_per_gpu
        auto_cast = _build_autocast(self.cfg)
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        logger.info(f"Fragment batch size: {batch_size_test}")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        # 创建 benchmark writer（用于竞赛提交格式，根据数据集类型自动选择）
        benchmark_writer = create_benchmark_writer(
            dataset_type=self.cfg.data.test.type,
            save_dir=self.cfg.save_path,
            dataset=self.test_loader.dataset,
        )
        if benchmark_writer is not None and comm.is_main_process():
            benchmark_writer.setup()
        # 创建通用 writer（用于 LAS/PLY/PCD 等实际生产输出，可选）
        general_writer = None
        writer_cfg = getattr(self.cfg, "writer", None)
        if writer_cfg is not None:
            general_writer = build_writer(writer_cfg)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
            num_fragments = len(fragment_list)
            batch_num = int(np.ceil(num_fragments / batch_size_test))
            for i in range(batch_num):
                s_i = i * batch_size_test
                e_i = min((i + 1) * batch_size_test, num_fragments)
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad(), auto_cast():
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Fragment: {frag_end}/{frag_total}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        frag_end=e_i,
                        frag_total=num_fragments,
                    )
                )
            if benchmark_writer is not None and benchmark_writer.topk > 1:
                pred = pred.topk(benchmark_writer.topk, dim=1)[1].data.cpu().numpy()
            else:
                pred = pred.max(1)[1].data.cpu().numpy()
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            # 写入 benchmark 提交文件
            if benchmark_writer is not None:
                benchmark_writer.write(data_name, pred)
                pred = benchmark_writer.pred_for_eval(pred)
            # 写入通用格式文件（可选）
            if general_writer is not None:
                general_writer.write(data_name, pred_sem=self._remap_to_original(pred))

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            self.cache_cleaner.check_and_clean(
                batch_time.val, f"test iter {idx + 1}/{len(self.test_loader)}"
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if benchmark_writer is not None:
                benchmark_writer.finalize(
                    intersection=intersection, union=union, target=target
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Test result:  mIoU {:.4f}  mAcc {:.4f}  OA {:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            names = self.cfg.data.names
            precision_class = intersection / (intersection + np.maximum(union - target, 0) + 1e-10)
            f1_class = 2 * precision_class * accuracy_class / (precision_class + accuracy_class + 1e-10)
            max_name_len = max(len(n) for n in names)
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "  Class {:2d} - {:<{w}} | IoU {:.4f}  Prec {:.4f}  Recall {:.4f}  F1 {:.4f}".format(
                        i, names[i], iou_class[i], precision_class[i], accuracy_class[i], f1_class[i],
                        w=max_name_len,
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class DINOSemSegTester(TesterBase):
    def test(self):
        batch_size_test = self.cfg.batch_size_test_per_gpu
        auto_cast = _build_autocast(self.cfg)
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        logger.info(f"Fragment batch size: {batch_size_test}")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        # 创建 benchmark writer（用于竞赛提交格式，根据数据集类型自动选择）
        benchmark_writer = create_benchmark_writer(
            dataset_type=self.cfg.data.test.type,
            save_dir=self.cfg.save_path,
            dataset=self.test_loader.dataset,
        )
        if benchmark_writer is not None and comm.is_main_process():
            benchmark_writer.setup()
        # 创建通用 writer（用于 LAS/PLY/PCD 等实际生产输出，可选）
        general_writer = None
        writer_cfg = getattr(self.cfg, "writer", None)
        if writer_cfg is not None:
            general_writer = build_writer(writer_cfg)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            dino_coord = data_dict.pop("dino_coord").cuda(non_blocking=True)
            dino_feat = data_dict.pop("dino_feat").cuda(non_blocking=True)
            dino_offset = data_dict.pop("dino_offset").cuda(non_blocking=True)
            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
            num_fragments = len(fragment_list)
            batch_num = int(np.ceil(num_fragments / batch_size_test))
            for i in range(batch_num):
                s_i = i * batch_size_test
                e_i = min((i + 1) * batch_size_test, num_fragments)
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                input_dict["dino_coord"] = dino_coord
                input_dict["dino_feat"] = dino_feat
                input_dict["dino_offset"] = dino_offset
                idx_part = input_dict["index"]
                with torch.no_grad(), auto_cast():
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Fragment: {frag_end}/{frag_total}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        frag_end=e_i,
                        frag_total=num_fragments,
                    )
                )
            if benchmark_writer is not None and benchmark_writer.topk > 1:
                pred = pred.topk(benchmark_writer.topk, dim=1)[1].data.cpu().numpy()
            else:
                pred = pred.max(1)[1].data.cpu().numpy()
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            # 写入 benchmark 提交文件
            if benchmark_writer is not None:
                benchmark_writer.write(data_name, pred)
                pred = benchmark_writer.pred_for_eval(pred)
            # 写入通用格式文件（可选）
            if general_writer is not None:
                general_writer.write(data_name, pred_sem=self._remap_to_original(pred))

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            self.cache_cleaner.check_and_clean(
                batch_time.val, f"test iter {idx + 1}/{len(self.test_loader)}"
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if benchmark_writer is not None:
                benchmark_writer.finalize(
                    intersection=intersection, union=union, target=target
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Test result:  mIoU {:.4f}  mAcc {:.4f}  OA {:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            names = self.cfg.data.names
            precision_class = intersection / (intersection + np.maximum(union - target, 0) + 1e-10)
            f1_class = 2 * precision_class * accuracy_class / (precision_class + accuracy_class + 1e-10)
            max_name_len = max(len(n) for n in names)
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "  Class {:2d} - {:<{w}} | IoU {:.4f}  Prec {:.4f}  Recall {:.4f}  F1 {:.4f}".format(
                        i, names[i], iou_class[i], precision_class[i], accuracy_class[i], f1_class[i],
                        w=max_name_len,
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()
        auto_cast = _build_autocast(self.cfg)

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad(), auto_cast():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )
            self.cache_cleaner.check_and_clean(
                batch_time.val, f"test iter {i + 1}/{len(self.test_loader)}"
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()
        auto_cast = _build_autocast(self.cfg)

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad(), auto_cast():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                )
            )
            self.cache_cleaner.check_and_clean(
                batch_time.val, f"test iter {idx + 1}/{len(self.test_loader)}"
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()
        auto_cast = _build_autocast(self.cfg)

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad(), auto_cast():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )
            self.cache_cleaner.check_and_clean(
                batch_time.val, f"test iter {idx + 1}/{len(self.test_loader)}"
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class InsSegTester(TesterBase):
    def __init__(
        self,
        segment_ignore_index,
        instance_ignore_index,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.valid_class_names = [
            self.cfg.data.names[i]
            for i in range(self.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def test(self):
        auto_cast = _build_autocast(self.cfg)
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        self.model.eval()
        scenes = {}

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_name = data_dict.pop("name")
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
            with torch.no_grad(), auto_cast():
                output_dict = self.model(data_dict)
                segment = data_dict["segment"]
                instance = data_dict["instance"]

                if "origin_coord" in data_dict.keys():
                    reverse, _ = pointops.knn_query(
                        1,
                        data_dict["coord"].float(),
                        data_dict["offset"].int(),
                        data_dict["origin_coord"].float(),
                        data_dict["origin_offset"].int(),
                    )
                    reverse = reverse.cpu().flatten().long()
                    output_dict["pred_masks"] = output_dict["pred_masks"][:, reverse]
                    segment = data_dict["origin_segment"]
                    instance = data_dict["origin_instance"]

                gt_instances, pred_instance = self.associate_instances(
                    output_dict, segment, instance
                )

            scenes[data_name] = dict(gt=gt_instances, pred=pred_instance)
            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )
            if self.cfg.data.test.type == "ScanNetPPDataset":
                self.write_scannetpp_results(
                    output_dict["pred_scores"],
                    output_dict["pred_masks"],
                    output_dict["pred_classes"],
                    data_name,
                )
            self.cache_cleaner.check_and_clean(
                batch_time.val, f"test iter {idx + 1}/{len(self.test_loader)}"
            )

        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)

        if comm.is_main_process():
            scenes = {}
            for _ in range(len(scenes_sync)):
                r = scenes_sync.pop()
                scenes.update(r)
                del r
            scenes = list(scenes.values())
            ap_scores = self.evaluate_matches(scenes)
            all_ap = ap_scores["all_ap"]
            all_ap_50 = ap_scores["all_ap_50%"]
            all_ap_25 = ap_scores["all_ap_25%"]
            logger.info(
                "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                    all_ap, all_ap_50, all_ap_25
                )
            )
            for i, label_name in enumerate(self.valid_class_names):
                ap = ap_scores["classes"][label_name]["ap"]
                ap_50 = ap_scores["classes"][label_name]["ap50%"]
                ap_25 = ap_scores["classes"][label_name]["ap25%"]
                logger.info(
                    "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                        idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def write_scannetpp_results(
        self,
        pred_scores,
        pred_masks,
        pred_classes,
        data_name,
    ):
        pred_scores[pred_scores < 0] = 0
        pred_scores[pred_scores >= 0] = 1

        save_dir = os.path.join(self.cfg.save_path, "result", "submit")
        mask_dir = os.path.join(save_dir, "predicted_masks")
        make_dirs(mask_dir)

        result_path = os.path.join(save_dir, f"{data_name}.txt")
        result_file = open(result_path, "w")
        for i, (score, mask, cls) in enumerate(
            zip(
                pred_scores.cpu().numpy(),
                pred_masks.cpu().numpy(),
                pred_classes.cpu().numpy(),
            )
        ):
            mask = mask.astype(np.uint8)
            length = mask.shape[0]
            mask = np.concatenate([[0], mask, [0]])
            runs = np.where(mask[1:] != mask[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            counts = " ".join(str(x) for x in runs)
            rle = dict(length=length, counts=counts)

            mask_path = os.path.join(mask_dir, f"{data_name}_{i:03d}.json")
            relative_path = os.path.join("predicted_masks", f"{data_name}_{i:03d}.json")
            with open(mask_path, "w") as mask_file:
                json.dump(rle, mask_file, indent=2)
            result_file.write(f"{relative_path} {cls} {score:.3f}\n")
        result_file.close()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    @staticmethod
    def collate_fn(batch):
        # Restrict to bs 1
        return batch[0]


@TESTERS.register_module()
class RegressionTester(TesterBase):
    """Test-time evaluator for per-point regression tasks.

    Follows the same fragment-based inference pattern as ``SemSegTester``:
    for every scene the test dataset yields a ``fragment_list``; each
    fragment is fed through the model and the raw ``reg_pred`` outputs
    are *summed* into an accumulator of shape ``(N,)`` (or ``(N, D)``
    for multi-target). After processing all fragments the accumulated
    values are divided by a per-point count to obtain the final
    averaged prediction.

    If the dataset supplies the regression target (popped into
    ``"regression_target"`` by ``prepare_test_data``), MAE / RMSE / R²
    are computed and logged.

    Results are saved as ``<data_name>_pred_reg.npy`` and optionally
    written via a general writer (``pred_reg`` kwarg).
    """

    def test(self):
        batch_size_test = self.cfg.batch_size_test_per_gpu
        auto_cast = _build_autocast(self.cfg)
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Regression Evaluation >>>>>>>>>>>>>>>>")
        logger.info(f"Fragment batch size: {batch_size_test}")

        batch_time = AverageMeter()
        self.model.eval()

        # Optional general writer (LAS / PLY / …)
        general_writer = None
        writer_cfg = getattr(self.cfg, "writer", None)
        if writer_cfg is not None:
            general_writer = build_writer(writer_cfg)

        comm.synchronize()
        record = {}

        # Determine number of regression targets
        num_targets = getattr(self.cfg.data, "num_targets", 1)

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # bs == 1
            fragment_list = data_dict.pop("fragment_list")
            # The target is popped by prepare_test_data into "regression_target"
            regression_target = data_dict.pop("regression_target", None)
            data_name = data_dict.pop("name")

            n_points = (
                regression_target.shape[0]
                if regression_target is not None
                else data_dict.get("origin_coord", data_dict.get("coord")).shape[0]
            )
            if num_targets > 1:
                pred_sum = torch.zeros((n_points, num_targets)).cuda()
            else:
                pred_sum = torch.zeros(n_points).cuda()
            pred_count = torch.zeros(n_points, dtype=torch.long).cuda()

            num_fragments = len(fragment_list)
            batch_num = int(np.ceil(num_fragments / batch_size_test))
            for i in range(batch_num):
                s_i = i * batch_size_test
                e_i = min((i + 1) * batch_size_test, num_fragments)
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad(), auto_cast():
                    pred_part = self.model(input_dict)["reg_pred"]  # (n,) or (n,D)
                # Accumulate by offset segments
                bs = 0
                for be in input_dict["offset"]:
                    pred_sum[idx_part[bs:be]] += pred_part[bs:be]
                    pred_count[idx_part[bs:be]] += 1
                    bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Fragment: {frag_end}/{frag_total}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        frag_end=e_i,
                        frag_total=num_fragments,
                    )
                )

            # Average over fragments
            pred_count = pred_count.clamp(min=1)
            if num_targets > 1:
                pred = (pred_sum / pred_count.unsqueeze(-1)).cpu().numpy()
            else:
                pred = (pred_sum / pred_count.float()).cpu().numpy()

            # Handle origin mapping (grid-subsampled back to original)
            if "origin_coord" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                if regression_target is not None and "origin_regression_target" in data_dict:
                    regression_target = data_dict.pop("origin_regression_target")

            # Writer output
            if general_writer is not None:
                general_writer.write(data_name, pred_reg=pred)

            # Metrics (if target available)
            if regression_target is not None:
                if isinstance(regression_target, torch.Tensor):
                    regression_target = regression_target.numpy()
                regression_target = regression_target.astype(np.float64).reshape(-1)
                pred_flat = pred.astype(np.float64).reshape(-1)
                diff = pred_flat - regression_target
                mae = np.mean(np.abs(diff))
                rmse = np.sqrt(np.mean(diff ** 2))
                ss_res = np.sum(diff ** 2)
                ss_tot = np.sum((regression_target - regression_target.mean()) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
                record[data_name] = dict(mae=mae, rmse=rmse, r2=r2, n=len(pred_flat))
            else:
                record[data_name] = dict(mae=None, rmse=None, r2=None, n=pred.shape[0])

            batch_time.update(time.time() - start)
            info = "Test: {} [{}/{}]-{} Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                data_name,
                idx + 1,
                len(self.test_loader),
                pred.shape[0],
                batch_time=batch_time,
            )
            if record[data_name]["mae"] is not None:
                info += " MAE {:.6f} RMSE {:.6f} R² {:.6f}".format(
                    record[data_name]["mae"],
                    record[data_name]["rmse"],
                    record[data_name]["r2"],
                )
            logger.info(info)
            self.cache_cleaner.check_and_clean(
                batch_time.val, f"test iter {idx + 1}/{len(self.test_loader)}"
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r

            has_targets = any(v["mae"] is not None for v in record.values())
            if has_targets:
                total_n = sum(v["n"] for v in record.values() if v["mae"] is not None)
                # Weighted average by point count
                w_mae = sum(
                    v["mae"] * v["n"] for v in record.values() if v["mae"] is not None
                ) / max(total_n, 1)
                w_rmse = sum(
                    v["rmse"] * v["n"]
                    for v in record.values()
                    if v["rmse"] is not None
                ) / max(total_n, 1)
                w_r2 = sum(
                    v["r2"] * v["n"] for v in record.values() if v["r2"] is not None
                ) / max(total_n, 1)
                logger.info(
                    "Val result: MAE {:.6f} | RMSE {:.6f} | R² {:.6f}".format(
                        w_mae, w_rmse, w_r2
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Regression Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class SemSegRegressionTester(TesterBase):
    """Test-time evaluator for joint semantic-segmentation + regression.

    Combines the fragment-based inference of ``SemSegTester`` and
    ``RegressionTester`` in a single pass: each fragment is forwarded
    through the model once, and both ``seg_logits`` and ``reg_pred``
    are accumulated.

    Outputs per scene:
        * ``<name>_pred.npy``     – semantic prediction (int)
        * ``<name>_pred_reg.npy`` – regression prediction (float)

    If ground-truth is available (``segment`` / ``regression_target``),
    mIoU and MAE / RMSE / R² are computed and logged.
    """

    def test(self):
        batch_size_test = self.cfg.batch_size_test_per_gpu
        auto_cast = _build_autocast(self.cfg)
        logger = get_root_logger()
        logger.info(
            ">>>>>>>>>>>>>>>> Start SemSeg-Regression Evaluation >>>>>>>>>>>>>>>>"
        )
        logger.info(f"Fragment batch size: {batch_size_test}")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        # Benchmark writer (seg only)
        benchmark_writer = create_benchmark_writer(
            dataset_type=self.cfg.data.test.type,
            save_dir=self.cfg.save_path,
            dataset=self.test_loader.dataset,
        )
        if benchmark_writer is not None and comm.is_main_process():
            benchmark_writer.setup()

        # General writer (LAS / PLY / …)
        general_writer = None
        writer_cfg = getattr(self.cfg, "writer", None)
        if writer_cfg is not None:
            general_writer = build_writer(writer_cfg)

        comm.synchronize()
        record = {}
        num_targets = getattr(self.cfg.data, "num_targets", 1)

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # bs == 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            regression_target = data_dict.pop("regression_target", None)
            data_name = data_dict.pop("name")

            # --- seg accumulator ---
            seg_collect = torch.zeros(
                (segment.size, self.cfg.data.num_classes)
            ).cuda()
            # --- reg accumulator ---
            n_points = (
                regression_target.shape[0]
                if regression_target is not None
                else segment.size
            )
            if num_targets > 1:
                reg_sum = torch.zeros((n_points, num_targets)).cuda()
            else:
                reg_sum = torch.zeros(n_points).cuda()
            reg_count = torch.zeros(n_points, dtype=torch.long).cuda()

            num_fragments = len(fragment_list)
            batch_num = int(np.ceil(num_fragments / batch_size_test))
            for i in range(batch_num):
                s_i = i * batch_size_test
                e_i = min((i + 1) * batch_size_test, num_fragments)
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad(), auto_cast():
                    output = self.model(input_dict)
                    seg_part = output["seg_logits"]   # (n, k)
                    seg_part = F.softmax(seg_part, -1)
                    reg_part = output["reg_pred"]      # (n,) or (n, D)

                bs = 0
                for be in input_dict["offset"]:
                    seg_collect[idx_part[bs:be], :] += seg_part[bs:be]
                    reg_sum[idx_part[bs:be]] += reg_part[bs:be]
                    reg_count[idx_part[bs:be]] += 1
                    bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Fragment: {frag_end}/{frag_total}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        frag_end=e_i,
                        frag_total=num_fragments,
                    )
                )

            # --- seg finalize ---
            if benchmark_writer is not None and benchmark_writer.topk > 1:
                seg_pred = (
                    seg_collect.topk(benchmark_writer.topk, dim=1)[1]
                    .data.cpu()
                    .numpy()
                )
            else:
                seg_pred = seg_collect.max(1)[1].data.cpu().numpy()

            # --- reg finalize ---
            reg_count = reg_count.clamp(min=1)
            if num_targets > 1:
                reg_pred = (reg_sum / reg_count.unsqueeze(-1)).cpu().numpy()
            else:
                reg_pred = (reg_sum / reg_count.float()).cpu().numpy()

            # Handle origin mapping
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                seg_pred = seg_pred[data_dict["inverse"]]
                reg_pred = reg_pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
                if (
                    regression_target is not None
                    and "origin_regression_target" in data_dict
                ):
                    regression_target = data_dict.pop("origin_regression_target")

            # Benchmark writer (seg)
            if benchmark_writer is not None:
                benchmark_writer.write(data_name, seg_pred)
                seg_pred = benchmark_writer.pred_for_eval(seg_pred)
            # General writer (seg + reg)
            if general_writer is not None:
                general_writer.write(
                    data_name,
                    pred_sem=self._remap_to_original(seg_pred),
                    pred_reg=reg_pred,
                )

            # --- seg metrics ---
            intersection, union, target_seg = intersection_and_union(
                seg_pred,
                segment,
                self.cfg.data.num_classes,
                self.cfg.data.ignore_index,
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target_seg)

            # --- reg metrics ---
            reg_info = dict(mae=None, rmse=None, r2=None, n=reg_pred.shape[0])
            if regression_target is not None:
                if isinstance(regression_target, torch.Tensor):
                    regression_target = regression_target.numpy()
                rt = regression_target.astype(np.float64).reshape(-1)
                rp = reg_pred.astype(np.float64).reshape(-1)
                diff = rp - rt
                reg_info["mae"] = np.mean(np.abs(diff))
                reg_info["rmse"] = np.sqrt(np.mean(diff ** 2))
                ss_res = np.sum(diff ** 2)
                ss_tot = np.sum((rt - rt.mean()) ** 2)
                reg_info["r2"] = 1.0 - ss_res / max(ss_tot, 1e-10)
            record[data_name] = dict(
                intersection=intersection,
                union=union,
                target=target_seg,
                **reg_info,
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target_seg) + 1e-10)
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - start)
            info = (
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Acc {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            if reg_info["mae"] is not None:
                info += " MAE {:.6f} R² {:.6f}".format(
                    reg_info["mae"], reg_info["r2"]
                )
            logger.info(info)
            self.cache_cleaner.check_and_clean(
                batch_time.val, f"test iter {idx + 1}/{len(self.test_loader)}"
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r

            # --- seg summary ---
            intersection = np.sum(
                [m["intersection"] for m in record.values()], axis=0
            )
            union = np.sum([m["union"] for m in record.values()], axis=0)
            target_seg = np.sum([m["target"] for m in record.values()], axis=0)

            if benchmark_writer is not None:
                benchmark_writer.finalize(
                    intersection=intersection, union=union, target=target_seg
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target_seg + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target_seg) + 1e-10)

            logger.info(
                "Test result:  mIoU {:.4f}  mAcc {:.4f}  OA {:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            names = self.cfg.data.names
            precision_class = intersection / (intersection + np.maximum(union - target_seg, 0) + 1e-10)
            f1_class = 2 * precision_class * accuracy_class / (precision_class + accuracy_class + 1e-10)
            max_name_len = max(len(n) for n in names)
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "  Class {:2d} - {:<{w}} | IoU {:.4f}  Prec {:.4f}  Recall {:.4f}  F1 {:.4f}".format(
                        i, names[i], iou_class[i], precision_class[i], accuracy_class[i], f1_class[i],
                        w=max_name_len,
                    )
                )

            # --- reg summary ---
            has_reg = any(v["mae"] is not None for v in record.values())
            if has_reg:
                total_n = sum(
                    v["n"] for v in record.values() if v["mae"] is not None
                )
                w_mae = (
                    sum(
                        v["mae"] * v["n"]
                        for v in record.values()
                        if v["mae"] is not None
                    )
                    / max(total_n, 1)
                )
                w_rmse = (
                    sum(
                        v["rmse"] * v["n"]
                        for v in record.values()
                        if v["rmse"] is not None
                    )
                    / max(total_n, 1)
                )
                w_r2 = (
                    sum(
                        v["r2"] * v["n"]
                        for v in record.values()
                        if v["r2"] is not None
                    )
                    / max(total_n, 1)
                )
                logger.info(
                    "Regression result: MAE {:.6f} | RMSE {:.6f} | R² {:.6f}".format(
                        w_mae, w_rmse, w_r2
                    )
                )

            logger.info(
                "<<<<<<<<<<<<<<<<< End SemSeg-Regression Evaluation <<<<<<<<<<<<<<<<<"
            )

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class CnfTester(TesterBase):
    """Test-time evaluator for Conditional Neural Field (CNF) tasks.

    Implements the two-phase *encode-once, query-many* inference:

    **Phase 1 – Encode**
        Run the backbone on all fragments of a scene, accumulating per-
        point features.  The final ``support_feat (N, C)`` and
        ``support_coord (N, 3)`` are cached on GPU.

    **Phase 2 – Dense query**
        Generate a regular XY grid (``query_resolution``) spanning the
        support bounding box.  Feed batches of ``(Q, 2)`` query
        coordinates through ``model.query_forward`` to obtain
        ``pred_z (Q,)``.

    **Phase 3 – Optional derivatives**
        If ``compute_derivatives`` is enabled, run the same query with
        ``requires_grad=True`` and extract slope / curvature via
        ``torch.autograd.grad``.

    Config-level attributes read from ``self.cfg``:
        query_resolution (float): Output grid spacing in coordinate
            units (default 0.5).
        query_batch_size (int): Maximum query points per forward pass
            (default 100 000).
        compute_derivatives (bool): Compute slope / curvature maps
            (default False).
    """

    def test(self):
        batch_size_test = self.cfg.batch_size_test_per_gpu
        query_resolution = getattr(self.cfg, "query_resolution", 0.5)
        query_batch_size = getattr(self.cfg, "query_batch_size", 100_000)
        compute_derivatives = getattr(self.cfg, "compute_derivatives", False)
        query_dim = getattr(self.cfg, "query_dim", 2)
        auto_cast = _build_autocast(self.cfg)
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start CNF Evaluation >>>>>>>>>>>>>>>>")
        logger.info(
            f"Fragment batch: {batch_size_test}  "
            f"Query resolution: {query_resolution}  "
            f"Query batch: {query_batch_size}  "
            f"Derivatives: {compute_derivatives}  "
            f"Query dim: {query_dim}"
        )

        batch_time = AverageMeter()
        self.model.eval()

        # Optional general writer (LAS / PLY / …)
        general_writer = None
        writer_cfg = getattr(self.cfg, "writer", None)
        if writer_cfg is not None:
            general_writer = build_writer(writer_cfg)

        comm.synchronize()

        # Unwrap DDP to access encode / interpolate / decode directly
        raw_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # batch_size == 1
            fragment_list = data_dict.pop("fragment_list")
            data_name = data_dict.pop("name")

            # ============================================================
            # Phase 1: Backbone encoding (run once per scene)
            # ============================================================
            support_feat_parts = []
            support_coord_parts = []
            num_fragments = len(fragment_list)
            frag_batch_num = int(np.ceil(num_fragments / batch_size_test))

            for i in range(frag_batch_num):
                s_i = i * batch_size_test
                e_i = min((i + 1) * batch_size_test, num_fragments)
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                with torch.no_grad(), auto_cast():
                    enc = raw_model.extract_feat(input_dict)

                # Map fragment indices back to original point space.
                # When GridSample is not used (CNF case: all support points
                # in one pass, no sub-sampling), there is no "index" key —
                # use an identity mapping so the averaging merge is a no-op.
                if "index" in input_dict:
                    frag_idx = input_dict["index"].cpu()
                else:
                    frag_idx = torch.arange(enc["feat"].shape[0])
                support_feat_parts.append(
                    (frag_idx, enc["feat"].cpu())
                )
                support_coord_parts.append(
                    (frag_idx, enc["coord"].cpu())
                )

                logger.info(
                    "CNF Encode: {}/{}-{}, Fragment: {}/{}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name,
                        e_i,
                        num_fragments,
                    )
                )

            # Merge fragments: average features at overlapping indices
            all_idx = torch.cat([p[0] for p in support_feat_parts])
            N = int(all_idx.max().item()) + 1
            C = support_feat_parts[0][1].shape[1]
            feat_sum = torch.zeros(N, C)
            feat_count = torch.zeros(N, 1)
            coord_sum = torch.zeros(N, 3)

            for frag_idx, frag_feat in support_feat_parts:
                feat_sum[frag_idx] += frag_feat
                feat_count[frag_idx] += 1
            for frag_idx, frag_coord in support_coord_parts:
                coord_sum[frag_idx] += frag_coord

            valid_mask = feat_count.squeeze(1) > 0
            feat_sum[valid_mask] /= feat_count[valid_mask]
            coord_sum[valid_mask] /= feat_count[valid_mask]

            # Handle inverse mapping (grid-subsampled → original cloud)
            if "inverse" in data_dict:
                inverse = data_dict["inverse"]
                support_feat = feat_sum[inverse].cuda()
                support_coord = coord_sum[inverse].cuda()
            else:
                support_feat = feat_sum[valid_mask].cuda()
                support_coord = coord_sum[valid_mask].cuda()

            logger.info(
                "CNF Encode done: {} support points, feat shape {}".format(
                    support_coord.shape[0], support_feat.shape
                )
            )

            # ============================================================
            # Phase 2: Build query grid (Dual Masking: Core BBox + Convex Hull)
            # ============================================================

            qd = query_dim

            # 1. 提取包含 Buffer 的全量点云坐标，用于计算物理凸包
            raw_coords_list = [frag["coord"] for frag in fragment_list]
            all_raw_coords = torch.cat(raw_coords_list, dim=0)
            hull_xy_np = all_raw_coords[:, :qd].cpu().numpy()

            # 2. 从 data_dict 中获取当前 Tile 的核心边界 (Core BBox)
            if "core_bbox" in data_dict:
                core_bbox = data_dict["core_bbox"]
                xy_min = torch.tensor(core_bbox[:qd], dtype=torch.float32)
                xy_max = torch.tensor(core_bbox[qd:qd * 2], dtype=torch.float32)
            else:
                # 兼容老数据：退回到实际包围盒
                xy_min = all_raw_coords[:, :qd].min(dim=0).values.cpu()
                xy_max = all_raw_coords[:, :qd].max(dim=0).values.cpu()

            # 3. 第一道掩码 (方刀)：仅在 Core BBox 内生成密集的初始网格
            axes = [
                torch.arange(
                    lo.item(),
                    hi.item() + query_resolution,  # 包含上限，确保相邻块无缝拼接
                    query_resolution,
                )
                for lo, hi in zip(xy_min, xy_max)
            ]
            grids = torch.meshgrid(*axes, indexing="ij")
            query_xy_full = torch.stack(
                [g.flatten() for g in grids], dim=1
            )  # (Q_full, 2)

            # 如果点数极少，直接返回方形网格
            if hull_xy_np.shape[0] < 4:
                query_xy = query_xy_full
                total_queries = query_xy.shape[0]
                logger.info(f"CNF Query grid: {total_queries} points (Fallback to Core BBox)")
            else:
                # 4. 第二道掩码 (剪刀)：使用全量点计算凸包，裁剪掉伸出真实边界的悬空网格
                try:
                    from scipy.spatial import ConvexHull
                    from matplotlib.path import Path

                    hull = ConvexHull(hull_xy_np)
                    hull_vertices = hull_xy_np[hull.vertices]

                    poly_path = Path(hull_vertices)
                    # radius 设置为分辨率的一半，防止边缘真实网格点被误杀
                    keep_mask = poly_path.contains_points(
                        query_xy_full.numpy(),
                        radius=query_resolution * 0.5,
                    )
                    query_xy = query_xy_full[keep_mask]

                except Exception as e:
                    logger.warning(f"Convex hull failed: {e}. Falling back to Core BBox grid.")
                    query_xy = query_xy_full

                total_queries = query_xy.shape[0]
                logger.info(
                    "CNF Query grid: {} points (core bbox {} -> hull masked {})".format(
                        total_queries,
                        query_xy_full.shape[0],
                        total_queries,
                    )
                )

            # ============================================================
            # Phase 3: Batch query through head
            # ============================================================
            pred_z_parts = []
            slope_parts = [] if compute_derivatives else None
            curvature_parts = [] if compute_derivatives else None

            for qi in range(0, total_queries, query_batch_size):
                q_batch = query_xy[qi : qi + query_batch_size].cuda()

                if compute_derivatives:
                    q_batch = q_batch.requires_grad_(True)

                ctx = (
                    torch.enable_grad()
                    if compute_derivatives
                    else torch.no_grad()
                )
                with ctx, auto_cast():
                    pz = raw_model.query_forward(
                        support_coord, support_feat, q_batch
                    )

                    if compute_derivatives:
                        # First-order: slope
                        grad = torch.autograd.grad(
                            pz.sum(), q_batch, create_graph=True
                        )[0]  # (B, qd)
                        slope = grad.norm(dim=1)  # (B,)
                        slope_parts.append(slope.detach().cpu())

                        # Second-order: curvature (Laplacian)
                        curv = torch.zeros(
                            q_batch.shape[0], device=q_batch.device
                        )
                        for d in range(qd):
                            g2 = torch.autograd.grad(
                                grad[:, d].sum(),
                                q_batch,
                                retain_graph=(d < qd - 1),
                            )[0]  # (B, qd)
                            curv = curv + g2[:, d]
                        curvature_parts.append(curv.detach().cpu())

                pred_z_parts.append(pz.detach().cpu())

                if (
                    (qi // query_batch_size + 1) % 10 == 0
                    or qi + query_batch_size >= total_queries
                ):
                    logger.info(
                        "CNF Query: {}/{}-{}, {}/{}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name,
                            min(qi + query_batch_size, total_queries),
                            total_queries,
                        )
                    )

            pred_z_all = torch.cat(pred_z_parts)  # (Q,)

            # ============================================================
            # Phase 4: Assemble output & write
            # ============================================================
            output_xyz = torch.column_stack(
                [
                    query_xy,
                    (
                        pred_z_all.unsqueeze(1)
                        if pred_z_all.dim() == 1
                        else pred_z_all
                    ),
                ]
            ).numpy()  # (Q, 3) or (Q, qd + num_targets)

            # Restore real-world coordinates if a shift was applied
            coord_shift = data_dict.get("coord_shift", None)
            if coord_shift is not None:
                coord_shift = np.asarray(coord_shift, dtype=np.float64)
                # output_xyz columns: [query_dims..., pred_targets...]
                # shift[:qd] applies to the XY query columns,
                # shift[qd:] applies to the Z (predicted) columns
                output_xyz[:, :qd] += coord_shift[:qd]
                n_targets = output_xyz.shape[1] - qd
                if coord_shift.shape[0] > qd and n_targets > 0:
                    # For CNF terrain: shift[2] (z_shift) adds back to pred_z
                    output_xyz[:, qd:qd + min(n_targets, coord_shift.shape[0] - qd)] += (
                        coord_shift[qd:qd + n_targets]
                    )
                logger.info(
                    "CNF: Restored real-world coords (shift={})".format(
                        np.array2string(coord_shift, precision=3)
                    )
                )

            write_kwargs = dict(pred_coord=output_xyz)
            if compute_derivatives and slope_parts:
                write_kwargs["slope"] = torch.cat(slope_parts).numpy()
            if compute_derivatives and curvature_parts:
                write_kwargs["curvature"] = (
                    torch.cat(curvature_parts).numpy()
                )

            if general_writer is not None:
                general_writer.write(data_name, **write_kwargs)

            batch_time.update(time.time() - start)
            logger.info(
                "CNF: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Support {n_support} Query {n_query}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    n_support=support_coord.shape[0],
                    n_query=total_queries,
                )
            )
            self.cache_cleaner.check_and_clean(
                batch_time.val,
                f"test iter {idx + 1}/{len(self.test_loader)}",
            )

        logger.info("<<<<<<<<<<<<<<<<< End CNF Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
