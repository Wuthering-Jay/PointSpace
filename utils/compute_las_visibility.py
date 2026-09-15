"""Write adaptive top-down visibility into LAS/LAZ point attributes.

The geometric definition is shared with ``utils/tile_las_image.py``. It uses
multiple shifted Z-buffers, a density-adaptive splat radius, and a return-order
prior to produce a continuous ``observability`` value in [0, 1]. The same
value is thresholded once and written into the standard ``classification``
field. No tiling, image, correspondence, or sidecar file is produced.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import laspy
import numpy as np
from tqdm import tqdm

try:
    from .las_io import write_standard_las
except ImportError:
    from las_io import write_standard_las

try:
    from pointspace.utils.surface_visibility import (
        SurfaceVisibilityResult,
        compute_surface_visibility,
    )
except ModuleNotFoundError as error:
    if error.name != "pointspace":
        raise
    # Direct/IDE execution puts ``utils`` rather than the repository root on
    # sys.path. Add the root without changing normal package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pointspace.utils.surface_visibility import (
        SurfaceVisibilityResult,
        compute_surface_visibility,
    )


try:
    from pointspace.utils.logger import get_root_logger
except ImportError:
    def get_root_logger():
        """Fallback logger when pointspace is not available."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger(__name__)


class LASVisibilityClassifier:
    """Write binary visibility and continuous observability to LAS/LAZ.

    ``classification`` stores ``visible_class`` or ``invisible_class``.
    A float32 ExtraBytes dimension named ``observability`` stores the shared
    continuous value. Existing dimensions and point order are preserved.

    Args:
        pointcloud_path: Input LAS/LAZ file or directory.
        output_path: Output file/directory. ``None`` updates inputs atomically.
        visible_class: Classification value assigned to visible points.
        invisible_class: Classification value assigned to occluded points.
        surface_cell_size: Nominal cell size; ``None``/``"auto"`` estimates it.
        surface_radius: Base splat radius; ``None``/``"auto"`` uses one cell.
        surface_z_tolerance: Gaussian height-gap scale in metres.
        reference_resolution: Optional orthophoto resolution lower bound.
        surface_oversample: Subdivide the nominal grid by this factor.
        surface_num_offsets: One grid or four half-cell shifted grids.
        surface_adaptive_radius: Adapt the splat radius to local first-return
            density.
        surface_max_radius_cells: Maximum discretized splat radius.
        surface_visibility_threshold: Threshold applied to observability.
        echo_decay: Return-order prior ``exp(-decay * (return_number - 1))``.
        surface_slope_compensation: Apply distance-aware height decay while
            splatting, preventing roof ridges from flattening pitched roofs.
        surface_max_slope_degrees: Maximum plausible surface slope used by
            the compensated splat.
        first_return_bonus: Bounded positive prior for first/single returns.
        recursive: Recursively scan descendant directories.
        num_workers: Number of complete files processed concurrently.
        overwrite: Allow replacing a separate existing output.
        skip_existing: Skip separate existing outputs.
    """

    def __init__(
        self,
        pointcloud_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        visible_class: int = 1,
        invisible_class: int = 0,
        surface_cell_size: Optional[Union[float, str]] = None,
        surface_radius: Optional[Union[float, str]] = None,
        surface_z_tolerance: float = 0.15,
        reference_resolution: float = 0.0,
        surface_oversample: int = 2,
        surface_num_offsets: int = 4,
        surface_adaptive_radius: bool = True,
        surface_max_radius_cells: int = 2,
        surface_visibility_threshold: float = 0.25,
        echo_decay: float = 0.7,
        surface_slope_compensation: bool = True,
        surface_max_slope_degrees: float = 60.0,
        first_return_bonus: float = 0.15,
        recursive: bool = False,
        num_workers: int = 1,
        overwrite: bool = False,
        skip_existing: bool = False,
    ):
        self.pointcloud_path = Path(pointcloud_path)
        self.output_path = Path(output_path) if output_path is not None else None
        self.visible_class = int(visible_class)
        self.invisible_class = int(invisible_class)
        self.surface_cell_size = surface_cell_size
        self.surface_radius = surface_radius
        self.surface_z_tolerance = float(surface_z_tolerance)
        self.reference_resolution = float(reference_resolution)
        self.surface_oversample = int(surface_oversample)
        self.surface_num_offsets = int(surface_num_offsets)
        self.surface_adaptive_radius = bool(surface_adaptive_radius)
        self.surface_max_radius_cells = int(surface_max_radius_cells)
        self.surface_visibility_threshold = float(surface_visibility_threshold)
        self.echo_decay = float(echo_decay)
        self.surface_slope_compensation = bool(surface_slope_compensation)
        self.surface_max_slope_degrees = float(surface_max_slope_degrees)
        self.first_return_bonus = float(first_return_bonus)
        self.recursive = bool(recursive)
        self.num_workers = num_workers
        self.overwrite = bool(overwrite)
        self.skip_existing = bool(skip_existing)
        self.logger = get_root_logger()

        self._validate_parameters()
        self.jobs = self._build_jobs()

    def _validate_parameters(self):
        if not self.pointcloud_path.exists():
            raise FileNotFoundError(
                f"Point-cloud path does not exist: {self.pointcloud_path}"
            )
        if self.pointcloud_path.is_file() and (
            self.pointcloud_path.suffix.lower() not in (".las", ".laz")
        ):
            raise ValueError("Input file must use the .las or .laz extension")
        for value, name in (
            (self.visible_class, "visible_class"),
            (self.invisible_class, "invisible_class"),
        ):
            if value < 0 or value > 31:
                raise ValueError(f"{name} must be in [0, 31]")
        if self.visible_class == self.invisible_class:
            raise ValueError("visible_class and invisible_class must differ")
        if not isinstance(self.num_workers, int) or isinstance(
            self.num_workers, bool
        ):
            raise TypeError("num_workers must be an integer")
        if self.num_workers < 1:
            raise ValueError("num_workers must be positive")

        # The shared function is the single source of truth for all surface
        # parameter validation. An empty cloud avoids any expensive work.
        compute_surface_visibility(
            np.empty((0, 3), dtype=np.float64),
            surface_cell_size=self.surface_cell_size,
            surface_radius=self.surface_radius,
            surface_z_tolerance=self.surface_z_tolerance,
            reference_resolution=self.reference_resolution,
            surface_oversample=self.surface_oversample,
            surface_num_offsets=self.surface_num_offsets,
            surface_adaptive_radius=self.surface_adaptive_radius,
            surface_max_radius_cells=self.surface_max_radius_cells,
            surface_visibility_threshold=self.surface_visibility_threshold,
            echo_decay=self.echo_decay,
            surface_slope_compensation=self.surface_slope_compensation,
            surface_max_slope_degrees=self.surface_max_slope_degrees,
            first_return_bonus=self.first_return_bonus,
        )

    def _find_files(self) -> List[Path]:
        if self.pointcloud_path.is_file():
            return [self.pointcloud_path]
        iterator = (
            self.pointcloud_path.rglob("*")
            if self.recursive
            else self.pointcloud_path.iterdir()
        )
        files = sorted(
            path for path in iterator
            if path.is_file() and path.suffix.lower() in (".las", ".laz")
        )
        if not files:
            raise FileNotFoundError(
                f"No LAS/LAZ files found in {self.pointcloud_path}"
            )
        return files

    def _build_jobs(self) -> List[Tuple[Path, Path]]:
        input_files = self._find_files()
        if self.output_path is None:
            return [(path, path) for path in input_files]
        if self.pointcloud_path.is_file():
            if self.output_path.suffix.lower() in (".las", ".laz"):
                return [(self.pointcloud_path, self.output_path)]
            return [(
                self.pointcloud_path,
                self.output_path / self.pointcloud_path.name,
            )]
        if self.output_path.suffix.lower() in (".las", ".laz"):
            raise ValueError("Directory input requires an output directory")
        return [
            (path, self.output_path / path.relative_to(self.pointcloud_path))
            for path in input_files
        ]

    def compute_visibility_state(
        self,
        points: np.ndarray,
        return_number: Optional[np.ndarray] = None,
        log_parameters: bool = True,
    ) -> SurfaceVisibilityResult:
        """Compute continuous and thresholded visibility with shared logic."""
        result = compute_surface_visibility(
            points=points,
            return_number=return_number,
            surface_cell_size=self.surface_cell_size,
            surface_radius=self.surface_radius,
            surface_z_tolerance=self.surface_z_tolerance,
            reference_resolution=self.reference_resolution,
            surface_oversample=self.surface_oversample,
            surface_num_offsets=self.surface_num_offsets,
            surface_adaptive_radius=self.surface_adaptive_radius,
            surface_max_radius_cells=self.surface_max_radius_cells,
            surface_visibility_threshold=self.surface_visibility_threshold,
            echo_decay=self.echo_decay,
            surface_slope_compensation=self.surface_slope_compensation,
            surface_max_slope_degrees=self.surface_max_slope_degrees,
            first_return_bonus=self.first_return_bonus,
        )
        if log_parameters:
            self.logger.info(
                f"  Adaptive surface Z-buffer: cell={result.cell_size:.3f}, "
                f"base_radius={result.base_radius:.3f}, "
                f"grid={result.width}x{result.height}, "
                f"offsets={self.surface_num_offsets}, "
                f"slope_comp={self.surface_slope_compensation}, "
                f"max_slope={self.surface_max_slope_degrees:.1f}deg, "
                f"first_bonus={self.first_return_bonus:.3f}, "
                f"threshold={self.surface_visibility_threshold:.3f}"
            )
        return result

    def compute_visibility(
        self,
        points: np.ndarray,
        return_number: Optional[np.ndarray] = None,
        log_parameters: bool = True,
    ) -> np.ndarray:
        """Compatibility method returning only the hard mask."""
        return self.compute_visibility_state(
            points,
            return_number=return_number,
            log_parameters=log_parameters,
        ).visible

    @staticmethod
    def _same_path(first: Path, second: Path) -> bool:
        return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
            os.path.abspath(second)
        )

    @staticmethod
    def _temporary_output_path(output_file: Path) -> Path:
        token = f"{os.getpid()}-{threading.get_ident()}"
        return output_file.with_name(
            f".{output_file.stem}.{token}.tmp{output_file.suffix}"
        )

    @staticmethod
    def _write_observability(las, observability: np.ndarray):
        """Create or update the float32 ExtraBytes observability dimension."""
        if "observability" not in las.point_format.dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name="observability",
                type=np.float32,
                description="surface observability",
            ))
        else:
            dimension = las.point_format.dimension_by_name("observability")
            if not np.issubdtype(dimension.dtype, np.floating):
                raise ValueError(
                    "Existing LAS dimension 'observability' must be floating "
                    f"point, got {dimension.dtype}"
                )
        las.observability = np.asarray(observability, dtype=np.float32)

    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        log_progress: bool = True,
    ) -> Dict:
        in_place = self._same_path(input_file, output_file)
        if output_file.exists() and not in_place:
            if self.skip_existing:
                return {
                    "input_file": input_file,
                    "output_file": output_file,
                    "status": "skipped",
                    "points": 0,
                    "visible_points": 0,
                }
            if not self.overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_file}. Set overwrite=True "
                    "or skip_existing=True."
                )

        start = time.time()
        if log_progress:
            self.logger.info(f"Processing {input_file}")
        las = laspy.read(input_file)
        points = np.column_stack((las.x, las.y, las.z))
        return_number = (
            np.asarray(las.return_number, dtype=np.float32)
            if "return_number" in las.point_format.dimension_names
            else None
        )
        result = self.compute_visibility_state(
            points,
            return_number=return_number,
            log_parameters=log_progress,
        )
        classification = np.full(
            len(points), self.invisible_class, dtype=np.uint8
        )
        classification[result.visible] = self.visible_class
        las.classification = classification
        self._write_observability(las, result.observability)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._temporary_output_path(output_file)
        try:
            converted_from_copc = write_standard_las(
                las, temporary_path, warn_on_copc=False
            )
            if converted_from_copc:
                self.logger.warning(
                    "COPC input %s was written as standard %s because laspy "
                    "does not support COPC writing. Point order, attributes "
                    "and CRS are preserved; COPC hierarchy indexing is removed.",
                    input_file,
                    output_file.suffix.upper().lstrip("."),
                )
            os.replace(temporary_path, output_file)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        visible_count = int(np.count_nonzero(result.visible))
        if log_progress:
            ratio = visible_count / len(points) if len(points) else 0.0
            self.logger.info(
                f"  Visibility: {visible_count:,}/{len(points):,} points "
                f"({ratio:.1%}); classification "
                f"visible={self.visible_class}, invisible={self.invisible_class}; "
                f"observability=float32; saved to {output_file} in "
                f"{time.time() - start:.2f}s"
            )
        return {
            "input_file": input_file,
            "output_file": output_file,
            "status": "updated",
            "points": len(points),
            "visible_points": visible_count,
            "converted_from_copc": converted_from_copc,
        }

    def process_all_files(self) -> Dict:
        worker_count = min(self.num_workers, max(1, len(self.jobs)))
        total_points = 0
        total_visible = 0
        updated_files = 0
        skipped_files = 0

        self.logger.info(
            f"Found {len(self.jobs)} LAS/LAZ files; workers={worker_count}; "
            f"mode={'in-place' if self.output_path is None else 'output'}"
        )
        with tqdm(
            total=len(self.jobs), desc="LAS visibility", unit="file"
        ) as progress:
            if worker_count == 1:
                for input_file, output_file in self.jobs:
                    result = self.process_file(
                        input_file, output_file, log_progress=False
                    )
                    total_points += result["points"]
                    total_visible += result["visible_points"]
                    updated_files += result["status"] == "updated"
                    skipped_files += result["status"] == "skipped"
                    progress.set_postfix_str(
                        f"visible={total_visible:,}/{total_points:,}"
                    )
                    progress.update()
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = {
                        executor.submit(
                            self.process_file, input_file, output_file, False
                        ): (input_file, output_file)
                        for input_file, output_file in self.jobs
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        total_points += result["points"]
                        total_visible += result["visible_points"]
                        updated_files += result["status"] == "updated"
                        skipped_files += result["status"] == "skipped"
                        progress.set_postfix_str(
                            f"visible={total_visible:,}/{total_points:,}"
                        )
                        progress.update()

        ratio = total_visible / total_points if total_points else 0.0
        self.logger.info(
            f"Visibility classification completed: {updated_files} updated, "
            f"{skipped_files} skipped, {total_visible:,}/{total_points:,} "
            f"visible points ({ratio:.1%})"
        )
        return {
            "files": len(self.jobs),
            "updated_files": updated_files,
            "skipped_files": skipped_files,
            "points": total_points,
            "visible_points": total_visible,
        }


def classify_las_visibility(
    pointcloud_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    visible_class: int = 1,
    invisible_class: int = 0,
    surface_cell_size: Optional[Union[float, str]] = None,
    surface_radius: Optional[Union[float, str]] = None,
    surface_z_tolerance: float = 0.15,
    reference_resolution: float = 0.0,
    surface_oversample: int = 2,
    surface_num_offsets: int = 4,
    surface_adaptive_radius: bool = True,
    surface_max_radius_cells: int = 2,
    surface_visibility_threshold: float = 0.25,
    echo_decay: float = 0.7,
    surface_slope_compensation: bool = True,
    surface_max_slope_degrees: float = 60.0,
    first_return_bonus: float = 0.15,
    recursive: bool = False,
    num_workers: int = 1,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict:
    """Classify one file or a directory of LAS/LAZ point clouds."""
    processor = LASVisibilityClassifier(
        pointcloud_path=pointcloud_path,
        output_path=output_path,
        visible_class=visible_class,
        invisible_class=invisible_class,
        surface_cell_size=surface_cell_size,
        surface_radius=surface_radius,
        surface_z_tolerance=surface_z_tolerance,
        reference_resolution=reference_resolution,
        surface_oversample=surface_oversample,
        surface_num_offsets=surface_num_offsets,
        surface_adaptive_radius=surface_adaptive_radius,
        surface_max_radius_cells=surface_max_radius_cells,
        surface_visibility_threshold=surface_visibility_threshold,
        echo_decay=echo_decay,
        surface_slope_compensation=surface_slope_compensation,
        surface_max_slope_degrees=surface_max_slope_degrees,
        first_return_bonus=first_return_bonus,
        recursive=recursive,
        num_workers=num_workers,
        overwrite=overwrite,
        skip_existing=skip_existing,
    )
    return processor.process_all_files()


if __name__ == "__main__":
    classify_las_visibility(
        # 输入可以是单个 LAS/LAZ，也可以是包含点云的文件夹。
        pointcloud_path=r"G:\data\swiss\swiss_cvrl_70km2\zurich\lidar\swisssurface3d_2024_2677-1244_2056_5728.copc.laz",

        # None 表示原地更新；如需保留原数据，可填写输出文件或目录。
        output_path=r"G:\data\swiss\swiss_cvrl_70km2\zurich\lidar1\swisssurface3d_2024_2677-1244_2056_5728.copc.laz",
        visible_class=1,
        invisible_class=0,

        # 建议先使用这组默认值，再按人工检查结果调整 threshold/decay。
        surface_cell_size="auto",
        surface_radius="auto",
        surface_z_tolerance=0.10,
        reference_resolution=0.0,
        surface_oversample=2,
        surface_num_offsets=4,
        surface_adaptive_radius=True,
        surface_max_radius_cells=2,
        surface_visibility_threshold=0.40,
        echo_decay=0.90,
        surface_slope_compensation=True,
        surface_max_slope_degrees=60.0,
        first_return_bonus=0.15,

        recursive=False,
        num_workers=1,
        overwrite=True,
        skip_existing=False,
    )
