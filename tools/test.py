"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import sys
from pathlib import Path
import argparse
# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pointspace.engines.defaults import (
    default_config_parser,
    default_setup,
)
from pointspace.engines.test import TESTERS
from pointspace.engines.launch import launch
from pointspace.utils.config import DictAction

def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default=r"configs\铁二院\semseg-litept-v1m1-0-base.py", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)",)
    parser.add_argument("--dist-url", default="auto",
        help="initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="custom options")

    return parser

def main_worker(cfg):
    cfg = default_setup(cfg)
    test_cfg = dict(cfg=cfg, **cfg.test)
    tester = TESTERS.build(test_cfg)
    tester.test()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
