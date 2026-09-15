"""Stream COPC files into conventional LAS/LAZ without dropping attributes.

The conversion preserves the LAS version, point format, scale/offset, CRS,
standard VLR/EVLR records, ExtraBytes definitions, point order and the packed
value of every point dimension. Only COPC hierarchy/index records are removed.

This module is intentionally configured from the ``__main__`` block for direct
IDE execution, matching the other PointSpace utilities.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Union

import laspy
import numpy as np
from tqdm import tqdm

try:
    from .las_io import _is_copc_record, standard_header_copy
except ImportError:
    from las_io import _is_copc_record, standard_header_copy


LOGGER = logging.getLogger("convert_copc")


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )


def _is_las_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".las", ".laz"}


def _header_is_copc(header) -> bool:
    records = list(header.vlrs) + list(header.evlrs or ())
    return any(_is_copc_record(record) for record in records)


def _standard_stem(path: Path) -> str:
    stem = path.stem
    if stem.lower().endswith(".copc"):
        stem = stem[:-5]
    return stem


def _discover_files(path: Path, recursive: bool) -> List[Path]:
    if _is_las_file(path):
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"LAS/LAZ input does not exist: {path}")
    iterator = path.rglob("*") if recursive else path.iterdir()
    files = sorted(candidate for candidate in iterator if _is_las_file(candidate))
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found in: {path}")
    return files


def _output_file(
    source: Path,
    input_path: Path,
    output_path: Path,
    output_format: str,
    preserve_structure: bool,
) -> Path:
    suffix = f".{output_format}"
    if input_path.is_file() and output_path.suffix.lower() in {".las", ".laz"}:
        return output_path
    relative_parent = (
        source.parent.relative_to(input_path)
        if input_path.is_dir() and preserve_structure
        else Path()
    )
    return output_path / relative_parent / f"{_standard_stem(source)}{suffix}"


def _point_bytes(points) -> bytes:
    return np.ascontiguousarray(points.array).tobytes()


def _dimension_names(header) -> Tuple[str, ...]:
    return tuple(str(name) for name in header.point_format.dimension_names)


def _verify_output(
    output_path: Path,
    expected_header,
    expected_count: int,
    expected_digest: str,
    chunk_size: int,
) -> None:
    digest = hashlib.sha256()
    point_count = 0
    with laspy.open(output_path, read_evlrs=True) as reader:
        header = reader.header
        if _header_is_copc(header):
            raise ValueError("output still contains COPC hierarchy records")
        if str(header.version) != str(expected_header.version):
            raise ValueError(
                f"LAS version changed: {expected_header.version} -> {header.version}"
            )
        if header.point_format.id != expected_header.point_format.id:
            raise ValueError(
                "point format changed: "
                f"{expected_header.point_format.id} -> {header.point_format.id}"
            )
        if _dimension_names(header) != _dimension_names(expected_header):
            raise ValueError("point dimension schema changed during conversion")
        if not np.array_equal(header.scales, expected_header.scales):
            raise ValueError("coordinate scales changed during conversion")
        if not np.array_equal(header.offsets, expected_header.offsets):
            raise ValueError("coordinate offsets changed during conversion")
        for points in reader.chunk_iterator(chunk_size):
            point_count += len(points)
            digest.update(_point_bytes(points))
    if point_count != expected_count:
        raise ValueError(f"point count changed: {expected_count} -> {point_count}")
    if digest.hexdigest() != expected_digest:
        raise ValueError("packed point attributes or point order changed")


def _convert_one(arguments) -> Dict:
    (
        source,
        destination,
        output_format,
        chunk_size,
        overwrite,
        copc_only,
        verify,
    ) = arguments
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output must be different files")
    if destination.exists() and not overwrite:
        return {"source": str(source), "output": str(destination), "status": "exists"}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    if temporary_path.exists():
        temporary_path.unlink()

    digest = hashlib.sha256()
    point_count = 0
    valid_return_points = 0
    has_return_fields = False
    try:
        with laspy.open(source, read_evlrs=True) as reader:
            source_header = reader.header
            is_copc = _header_is_copc(source_header)
            if copc_only and not is_copc:
                return {
                    "source": str(source),
                    "output": None,
                    "status": "not_copc",
                }
            output_header = standard_header_copy(source_header)
            dimension_names = set(_dimension_names(source_header))
            has_return_fields = {
                "return_number",
                "number_of_returns",
            }.issubset(dimension_names)
            with laspy.open(
                temporary_path,
                mode="w",
                header=output_header,
                do_compress=output_format == "laz",
            ) as writer:
                for points in reader.chunk_iterator(chunk_size):
                    point_count += len(points)
                    digest.update(_point_bytes(points))
                    if has_return_fields:
                        return_number = np.asarray(points.return_number)
                        number_of_returns = np.asarray(points.number_of_returns)
                        valid_return_points += int(np.count_nonzero(
                            (return_number >= 1)
                            & (number_of_returns >= 1)
                            & (return_number <= number_of_returns)
                        ))
                    writer.write_points(points)
                if output_header.evlrs:
                    writer.write_evlrs(output_header.evlrs)

        if verify:
            _verify_output(
                temporary_path,
                output_header,
                point_count,
                digest.hexdigest(),
                chunk_size,
            )
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "source": str(source),
        "output": str(destination),
        "status": "converted",
        "points": point_count,
        "valid_return_points": valid_return_points if has_return_fields else None,
        "invalid_return_points": (
            point_count - valid_return_points if has_return_fields else None
        ),
        "verified": verify,
    }


def convert_copc_to_standard_las(
    pointcloud_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: str = "laz",
    recursive: bool = True,
    preserve_structure: bool = True,
    chunk_size: int = 1_000_000,
    num_workers: int = 1,
    overwrite: bool = False,
    copc_only: bool = True,
    verify: bool = True,
) -> List[Dict]:
    """Convert one file or a directory tree to conventional LAS/LAZ.

    Conversion is streaming and atomic. ``verify=True`` re-reads each output
    and checks its packed point records byte-for-byte, which proves that every
    standard and ExtraBytes point attribute was retained. Verification roughly
    doubles I/O time but is recommended for the first conversion run.
    """
    _configure_logging()
    input_path = Path(pointcloud_path)
    output_path = Path(output_path)
    output_format = output_format.lower().lstrip(".")
    if output_format not in {"las", "laz"}:
        raise ValueError("output_format must be 'las' or 'laz'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")

    sources = _discover_files(input_path, recursive)
    tasks = []
    destinations = set()
    for source in sources:
        destination = _output_file(
            source,
            input_path,
            output_path,
            output_format,
            preserve_structure,
        )
        destination_key = str(destination.resolve()).lower()
        if destination_key in destinations:
            raise ValueError(
                f"Multiple inputs map to the same output: {destination}. "
                "Use preserve_structure=True."
            )
        destinations.add(destination_key)
        tasks.append((
            source,
            destination,
            output_format,
            chunk_size,
            overwrite,
            copc_only,
            verify,
        ))

    LOGGER.info(
        "Found %d LAS/LAZ files; workers=%d; format=%s; verify=%s",
        len(tasks),
        num_workers,
        output_format,
        verify,
    )
    results = []
    if num_workers == 1:
        for task in tqdm(tasks, desc="Convert COPC", unit="file"):
            results.append(_convert_one(task))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_convert_one, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Convert COPC",
                unit="file",
            ):
                results.append(future.result())

    converted = [result for result in results if result["status"] == "converted"]
    invalid_returns = sum(
        result.get("invalid_return_points") or 0 for result in converted
    )
    LOGGER.info(
        "Done: converted=%d, existing=%d, not_copc=%d, invalid_returns=%d",
        len(converted),
        sum(result["status"] == "exists" for result in results),
        sum(result["status"] == "not_copc" for result in results),
        invalid_returns,
    )
    if invalid_returns:
        LOGGER.warning(
            "%d points have invalid/missing return metadata in the source. "
            "Container conversion preserves these values and cannot create "
            "physically valid RRCL targets.",
            invalid_returns,
        )
    return results


if __name__ == "__main__":
    convert_copc_to_standard_las(
        pointcloud_path=r"G:\data\swiss\swiss_cvrl_70km2\zurich\lidar",
        output_path=r"G:\data\swiss\swiss_cvrl_70km2\zurich\lidar_standard",
        output_format="laz",
        recursive=True,
        preserve_structure=True,
        chunk_size=1_000_000,
        num_workers=1,
        overwrite=False,
        copc_only=True,
        verify=True,
    )
