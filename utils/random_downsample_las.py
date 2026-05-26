import argparse
import logging
from pathlib import Path
from typing import Optional

import laspy
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LASRandomDownsampler:
    """
    LAS/LAZ 点云随机抽稀工具。

    功能：
    1. 支持单个 LAS/LAZ 文件抽稀。
    2. 支持文件夹内 LAS/LAZ 批量抽稀。
    3. 保留所有原始点属性。
    4. 更新头文件中与点数、边界框相关的统计信息。
    """

    def __init__(self, divisor: int, seed: Optional[int] = None):
        if divisor <= 0:
            raise ValueError("divisor 必须为正整数")

        self.divisor = divisor
        self.rng = np.random.default_rng(seed)

    def downsample_path(self, input_path: str, output_path: str):
        in_path = Path(input_path)
        out_path = Path(output_path)

        if not in_path.exists():
            raise FileNotFoundError(f"输入路径不存在: {in_path}")

        if in_path.is_file():
            self._process_single_path(in_path, out_path)
            logging.info("所有抽稀任务已完成")
            return

        if not in_path.is_dir():
            raise ValueError(f"无效输入路径: {in_path}")

        out_path.mkdir(parents=True, exist_ok=True)
        las_files = [
            file_path for file_path in in_path.rglob("*")
            if file_path.suffix.lower() in [".las", ".laz"]
        ]

        if not las_files:
            logging.warning("输入文件夹中未找到 LAS/LAZ 文件")
            return

        for file_path in las_files:
            rel_path = file_path.relative_to(in_path)
            dst_path = out_path / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            self._downsample_file(file_path, dst_path)

        logging.info("所有抽稀任务已完成")

    def _process_single_path(self, input_file: Path, output_path: Path):
        if input_file.suffix.lower() not in [".las", ".laz"]:
            raise ValueError("输入文件必须是 .las 或 .laz 格式")

        if output_path.exists() and output_path.is_dir():
            output_file = output_path / input_file.name
        elif output_path.suffix.lower() in [".las", ".laz"]:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / input_file.name

        self._downsample_file(input_file, output_file)

    def _downsample_file(self, input_file: Path, output_file: Path):
        logging.info(f"正在处理: {input_file}")

        las = laspy.read(input_file)
        num_points = len(las.points)

        if num_points == 0:
            logging.warning(f"  -> 文件为空，直接复制为空输出: {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            las.update_header()
            las.write(output_file)
            return

        keep_count = max(1, num_points // self.divisor)

        if keep_count >= num_points:
            indices = np.arange(num_points)
        else:
            indices = self.rng.choice(num_points, size=keep_count, replace=False)
            indices.sort()

        # 直接切片 points，可完整保留原始点格式中的所有维度与 extra dims。
        las.points = las.points[indices]

        # 重新计算头文件中的点数、边界框等统计信息。
        las.update_header()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        las.write(output_file)

        logging.info(
            f"  -> 已保存至: {output_file} "
            f"(保留点数: {keep_count} / 原始点数: {num_points}, 抽稀倍率: 1/{self.divisor})"
        )


def build_parser():
    parser = argparse.ArgumentParser(description="LAS/LAZ 点云随机抽稀工具")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入 LAS/LAZ 文件路径，或包含点云的文件夹路径",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出文件路径或输出文件夹路径",
    )
    parser.add_argument(
        "-n", "--divisor",
        type=int,
        required=True,
        help="抽稀倍率 n，表示将点云随机抽稀为原来的 1/n",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；设置后可复现抽稀结果",
    )
    return parser


def main():
    input=r"E:\data\DALES\dales_las\tile\test"
    output=r"E:\data\DALES\dales_las\tile\test16"
    divisor=16
    seed=42
    downsampler = LASRandomDownsampler(divisor=divisor, seed=seed)
    downsampler.downsample_path(input, output)


if __name__ == "__main__":
    main()
