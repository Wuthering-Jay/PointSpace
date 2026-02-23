"""
Writer Builder

点云结果写入器的注册表与构建函数。
遵循 pointspace 的 Registry 模式（与 MODELS, TESTERS 等一致），
通过 WRITERS 注册表实现 Writer 的注册与构建。

Usage:
    writer = build_writer(dict(type="LASWriter", save_dir="output/"))
"""

import copy
from pointspace.utils.registry import Registry

WRITERS = Registry("writers")


def build_writer(cfg):
    """
    根据配置字典构建 Writer 实例。

    Args:
        cfg (dict): 必须包含 "type" 字段，其余字段作为构造参数传入。
            示例: dict(type="LASWriter", save_dir="output/", source_dir="data/raw/")

    Returns:
        BaseWriter: 构建好的 Writer 实例。
    """
    cfg = copy.deepcopy(cfg)
    writer_type = cfg.pop("type")
    writer_cls = WRITERS.get(writer_type)
    if writer_cls is None:
        raise KeyError(f"'{writer_type}' is not in the WRITERS registry. "
                       f"Available: {list(WRITERS._module_dict.keys())}")
    return writer_cls(**cfg)
