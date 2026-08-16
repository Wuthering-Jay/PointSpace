"""
Default training/testing logic

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import argparse
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


import pointspace.utils.comm as comm
from pointspace.utils.env import get_random_seed, set_seed
from pointspace.utils.config import Config, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = None if seed is None else num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser


def default_config_parser(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    # loop is set explicitly in each dataset config; no auto-override here.

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size

    # ---- batch size: prefer task-specific names and retain legacy fallback ----
    # ``batch_size`` is deprecated.  When an old config still uses it, publish
    # the resolved value as ``batch_size_train`` because Trainer consistently
    # consumes the task-specific name after setup.
    batch_size_train = getattr(cfg, "batch_size_train", None)
    if batch_size_train is None:
        batch_size_train = getattr(cfg, "batch_size", None)
    if batch_size_train is None:
        raise ValueError("Config must define batch_size_train")
    if batch_size_train <= 0 or batch_size_train % world_size != 0:
        raise ValueError(
            "batch_size_train must be positive and divisible by world_size "
            f"({world_size}), got {batch_size_train}"
        )
    cfg.batch_size_train = batch_size_train
    cfg.batch_size_train_per_gpu = batch_size_train // world_size

    # ---- micro-batch: gradient accumulation reduces per-step batch ----
    # batch_size_per_gpu is the ACTUAL DataLoader batch size (micro-batch).
    # Effective batch = micro-batch × gradient_accumulation_steps.
    grad_accum = getattr(cfg, "gradient_accumulation_steps", 1)
    if grad_accum <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    cfg.batch_size_per_gpu = max(1, cfg.batch_size_train_per_gpu // grad_accum)

    # Validation and test sizes are optional.  Missing/None means one sample
    # (or one test fragment for fragment-based testers) per GPU.
    batch_size_val = getattr(cfg, "batch_size_val", None)
    batch_size_test = getattr(cfg, "batch_size_test", None)
    if batch_size_val is not None and (
        batch_size_val <= 0 or batch_size_val % world_size != 0
    ):
        raise ValueError(
            "batch_size_val must be positive and divisible by world_size "
            f"({world_size}), got {batch_size_val}"
        )
    if batch_size_test is not None and (
        batch_size_test <= 0 or batch_size_test % world_size != 0
    ):
        raise ValueError(
            "batch_size_test must be positive and divisible by world_size "
            f"({world_size}), got {batch_size_test}"
        )
    cfg.batch_size_val_per_gpu = (
        batch_size_val // world_size if batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        batch_size_test // world_size if batch_size_test is not None else 1
    )
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed + rank * cfg.num_worker_per_gpu
    set_seed(seed)
    return cfg
