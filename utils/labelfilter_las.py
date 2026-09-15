"""Filter LAS/LAZ points by classification."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import laspy
import numpy as np
from tqdm import tqdm

try:
    from .las_io import write_standard_las
except ImportError:
    from las_io import write_standard_las


def _normalize_class_filters(
    classes_to_keep: Optional[Sequence[int]],
    classes_to_remove: Optional[Sequence[int]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Validate and normalize the mutually exclusive class filters."""

    def normalize(values, name):
        if values is None:
            return ()
        normalized = tuple(dict.fromkeys(int(value) for value in values))
        if any(value < 0 or value > 255 for value in normalized):
            raise ValueError(f"{name} values must be in [0, 255]")
        return normalized

    keep = normalize(classes_to_keep, "classes_to_keep")
    remove = normalize(classes_to_remove, "classes_to_remove")
    if bool(keep) == bool(remove):
        raise ValueError(
            "Exactly one of classes_to_keep and classes_to_remove must be non-empty"
        )
    return keep, remove


def filter_single_las(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    classes_to_keep: Optional[Sequence[int]] = None,
    classes_to_remove: Optional[Sequence[int]] = None,
    verbose: bool = True,
):
    """Filter one LAS/LAZ using either an inclusion or exclusion class list."""
    input_file = Path(input_file)
    output_file = Path(output_file)
    keep, remove = _normalize_class_filters(classes_to_keep, classes_to_remove)

    if verbose:
        print(f"正在处理: {input_file} ...")
    las = laspy.read(input_file)
    classification = np.asarray(las.classification)
    if keep:
        mask = np.isin(classification, keep)
        operation = f"保留类别 {list(keep)}"
    else:
        mask = ~np.isin(classification, remove)
        operation = f"去除类别 {list(remove)}"

    selected_count = int(np.count_nonzero(mask))
    if selected_count == 0 and verbose:
        print(
            f"  -> 警告: {input_file.name} 在{operation}后没有剩余点，"
            "将输出空文件！"
        )

    las.points = las.points[mask]
    las.update_header()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_standard_las(las, output_file)
    if verbose:
        print(
            f"  -> 已保存至: {output_file} "
            f"(保留点数: {selected_count} / 原始点数: {len(mask)})"
        )
    return {
        "input_file": input_file,
        "output_file": output_file,
        "original_points": len(mask),
        "selected_points": selected_count,
    }


def extract_las_by_class(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    classes_to_keep: Optional[Sequence[int]] = None,
    classes_to_remove: Optional[Sequence[int]] = None,
    recursive: bool = False,
    num_workers: Optional[int] = None,
):
    """Filter one file or all LAS/LAZ files in a directory.

    Exactly one of ``classes_to_keep`` and ``classes_to_remove`` must be
    non-empty. With ``recursive=True``, descendant directories are scanned and
    their relative directory structure is preserved below ``output_path``.
    ``num_workers`` controls concurrent file-level processing; by default up to
    eight workers are used to avoid excessive memory usage for large LAZ files.
    """
    keep, remove = _normalize_class_filters(classes_to_keep, classes_to_remove)
    input_path = Path(input_path)
    output_path = Path(output_path)
    if num_workers is None:
        num_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
    if not isinstance(num_workers, int) or isinstance(num_workers, bool):
        raise TypeError("num_workers must be an integer or None")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in (".las", ".laz"):
            raise ValueError("输入文件必须是 .las 或 .laz 格式")
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / input_path.name
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path
        jobs = [(input_path, output_file)]

    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        iterator = input_path.rglob("*") if recursive else input_path.iterdir()
        input_files = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in (".las", ".laz")
        )
        jobs = []
        for input_file in input_files:
            relative_path = input_file.relative_to(input_path)
            output_file = output_path / relative_path
            jobs.append((input_file, output_file))

    worker_count = min(num_workers, max(1, len(jobs)))
    total_original = 0
    total_selected = 0
    with tqdm(total=len(jobs), desc="Filter LAS/LAZ", unit="file") as progress:
        if worker_count == 1:
            for input_file, output_file in jobs:
                result = filter_single_las(
                    input_file, output_file, keep, remove, verbose=False
                )
                total_original += result["original_points"]
                total_selected += result["selected_points"]
                progress.update()
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        filter_single_las,
                        input_file,
                        output_file,
                        keep,
                        remove,
                        False,
                    )
                    for input_file, output_file in jobs
                ]
                for future in as_completed(futures):
                    result = future.result()
                    total_original += result["original_points"]
                    total_selected += result["selected_points"]
                    progress.update()

    print(
        f"所有过滤提取任务已完成：{len(jobs)} 个文件，"
        f"{total_selected:,}/{total_original:,} 个点已保留。"
    )
    return {
        "files": len(jobs),
        "original_points": total_original,
        "selected_points": total_selected,
    }



if __name__ == "__main__":
    extract_las_by_class(
        input_path=r"G:\data\Luxembourg\luxembourg2019_cvrl\lidar_laz",
        output_path=r"G:\data\Luxembourg\luxembourg2019_cvrl\lidar_laz_no_noise",
        classes_to_keep=None,
        classes_to_remove=[7, 18],
        recursive=True,
        num_workers=8,
    )
