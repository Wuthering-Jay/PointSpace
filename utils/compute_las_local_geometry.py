"""Append offline normal and local-shape supervision to ordinary LAS/LAZ files.

Edit the parameters in ``__main__`` and run directly in the pointspace IDE
environment. Output files retain every original point/attribute; ten new LAS
extra dimensions are appended. The input directory is never modified.

Normals and descriptors use K nearest 0.25 m voxel-centroid reference points
by default, approximating the spatial scale of Swiss-100 GridSample. This is
an offline, deterministic target: later random crop/dropout/jitter changes the
current point set, so this is not exactly the online post-augmentation KNN.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from uuid import uuid4

import laspy
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

LOGGER = logging.getLogger("las_local_geometry")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DESCRIPTOR_NAMES = (
    "geom_linearity", "geom_planarity", "geom_scattering",
    "geom_curvature", "geom_anisotropy", "geom_verticality",
)
NORMAL_NAMES = ("normal_x", "normal_y", "normal_z")
VALID_NAME = "geom_valid"


def voxel_centroids(xyz: np.ndarray, grid_size: float) -> np.ndarray:
    """Deterministic reference points, with no label/return input."""
    if grid_size <= 0:
        return np.asarray(xyz, dtype=np.float64)
    cell = np.floor((xyz - xyz.min(axis=0)) / grid_size).astype(np.int64)
    extent = cell.max(axis=0) + 1
    if int(extent[0]) * int(extent[1]) * int(extent[2]) < np.iinfo(np.int64).max:
        # 1-D keys are substantially cheaper than np.unique(axis=0) for
        # million-point LAS tiles and retain the same lexicographic ordering.
        key = (cell[:, 0] * extent[1] + cell[:, 1]) * extent[2] + cell[:, 2]
        _, inverse = np.unique(key, return_inverse=True)
    else:
        _, inverse = np.unique(cell, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    result = np.empty((count.size, 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.bincount(inverse, weights=xyz[:, axis]) / count
    return result


def compute_local_geometry(
    xyz: np.ndarray,
    *,
    k: int = 48,
    reference_grid_size: float = 0.25,
    chunk_size: int = 8192,
    min_neighbors: int = 8,
    min_variance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [N,3] normals, [N,6] descriptors and [N] valid mask.

    Eigenvalues are ordered λ1≥λ2≥λ3. Descriptor columns follow
    ``DESCRIPTOR_NAMES``. Verticality is ``1−abs(normal_z)``. Normals are
    oriented toward +Z where possible, but training should use a
    sign-invariant cosine loss for near-vertical surfaces.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be finite [N,3]")
    if k < 3 or chunk_size < 1 or min_neighbors < 3:
        raise ValueError("k, chunk_size and min_neighbors must be valid")
    n = xyz.shape[0]
    normals = np.zeros((n, 3), np.float32)
    shape = np.zeros((n, 6), np.float32)
    valid = np.zeros(n, bool)
    if n == 0:
        return normals, shape, valid
    reference = voxel_centroids(xyz, reference_grid_size)
    count = min(k, reference.shape[0])
    if count < min_neighbors:
        return normals, shape, valid
    tree = cKDTree(reference)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        _, neighbor = tree.query(xyz[start:stop], k=count, workers=1)
        points = reference[np.asarray(neighbor).reshape(stop - start, count)]
        centered = points - points.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / count
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        eigenvalue = np.maximum(eigenvalue, 0.0)
        lam3, lam2, lam1 = (eigenvalue[:, i] for i in (0, 1, 2))
        good = (lam1 > min_variance) & np.isfinite(eigenvalue).all(axis=1)
        normal = eigenvector[:, :, 0]
        normal[normal[:, 2] < 0] *= -1
        denominator = np.maximum(lam1, min_variance)
        total = np.maximum(lam1 + lam2 + lam3, min_variance)
        descriptors = np.column_stack((
            (lam1 - lam2) / denominator,
            (lam2 - lam3) / denominator,
            lam3 / denominator,
            lam3 / total,
            (lam1 - lam3) / denominator,
            1.0 - np.abs(normal[:, 2]),
        ))
        normals[start:stop] = np.where(good[:, None], normal, 0).astype(np.float32)
        shape[start:stop] = np.where(good[:, None], np.clip(descriptors, 0, 1), 0).astype(np.float32)
        valid[start:stop] = good
    return normals, shape, valid


def compute_local_geometry_gpu(
    xyz: np.ndarray,
    *,
    k: int = 48,
    reference_grid_size: float = 0.25,
    chunk_size: int = 8192,
    min_neighbors: int = 8,
    min_variance: float = 1e-10,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same targets as the CPU path using pointops KNN and CUDA 3x3 PCA.

    Coordinates are translated before conversion to float32; projected Swiss
    coordinates would otherwise lose meaningful sub-meter precision. Pointops
    queries the voxel-centroid reference set for each original LAS point.
    At most two file workers should share one GPU, allowing LAS I/O to overlap
    while keeping GPU memory and CUDA batched eigensolver concurrency bounded.
    """
    import torch
    import pointops

    if not torch.cuda.is_available():
        raise RuntimeError("GPU geometry backend requires CUDA")
    if not hasattr(pointops, "knn_query_idx"):
        raise RuntimeError("Installed pointops lacks knn_query_idx")
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be finite [N,3]")
    if k < 3 or chunk_size < 1 or min_neighbors < 3:
        raise ValueError("k, chunk_size and min_neighbors must be valid")
    # cuSolver's 3x3 batched eigh rejects very large batches on the tested
    # Windows/CUDA build even when tensors are finite and memory is available.
    if chunk_size > 16384:
        LOGGER.warning("Capping GPU chunk_size=%d at cuSolver-safe 16384", chunk_size)
        chunk_size = 16384
    n = xyz.shape[0]
    normals = np.zeros((n, 3), np.float32)
    shape = np.zeros((n, 6), np.float32)
    valid = np.zeros(n, bool)
    if n == 0:
        return normals, shape, valid
    reference = voxel_centroids(xyz, reference_grid_size)
    count = min(k, reference.shape[0])
    if count < min_neighbors:
        return normals, shape, valid
    origin = xyz.min(axis=0)
    reference_gpu = torch.from_numpy(
        np.ascontiguousarray((reference - origin).astype(np.float32))
    ).to(device)
    reference_offset = torch.tensor([reference.shape[0]], dtype=torch.int32, device=device)
    with torch.inference_mode():
        start = 0
        while start < n:
            stop = min(start + chunk_size, n)
            try:
                query_gpu = torch.from_numpy(
                    np.ascontiguousarray((xyz[start:stop] - origin).astype(np.float32))
                ).to(device)
                query_offset = torch.tensor([stop - start], dtype=torch.int32, device=device)
                neighbor = pointops.knn_query_idx(
                    count, reference_gpu, reference_offset, query_gpu, query_offset
                )
                points = reference_gpu[neighbor.long()]
                centered = points - points.mean(dim=1, keepdim=True)
                covariance = torch.bmm(centered.transpose(1, 2), centered) / count
                eigenvalue, eigenvector = torch.linalg.eigh(covariance)
                eigenvalue = eigenvalue.clamp_min(0)
                lam3, lam2, lam1 = eigenvalue.unbind(dim=1)
                good = (lam1 > min_variance) & torch.isfinite(eigenvalue).all(dim=1)
                normal = eigenvector[:, :, 0]
                normal = torch.where(normal[:, 2:3] < 0, -normal, normal)
                denominator = lam1.clamp_min(min_variance)
                total = (lam1 + lam2 + lam3).clamp_min(min_variance)
                descriptors = torch.stack((
                    (lam1 - lam2) / denominator,
                    (lam2 - lam3) / denominator,
                    lam3 / denominator,
                    lam3 / total,
                    (lam1 - lam3) / denominator,
                    1 - normal[:, 2].abs(),
                ), dim=1).clamp_(0, 1)
                normals[start:stop] = torch.where(good[:, None], normal, 0).cpu().numpy()
                shape[start:stop] = torch.where(good[:, None], descriptors, 0).cpu().numpy()
                valid[start:stop] = good.cpu().numpy()
                start = stop
                del query_gpu, query_offset, neighbor, points, centered, covariance
            except torch.cuda.OutOfMemoryError:
                if chunk_size <= 1024:
                    raise
                chunk_size = max(1024, chunk_size // 2)
                LOGGER.warning("CUDA OOM; reducing geometry chunk_size to %d", chunk_size)
                torch.cuda.empty_cache()
    return normals, shape, valid


def _input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [p for p in (path.rglob("*") if recursive else path.iterdir()) if p.is_file()]
    else:
        raise FileNotFoundError(path)
    result = sorted(p for p in files if p.suffix.lower() in {".las", ".laz"}
                    and not p.name.startswith("."))
    if not result:
        raise FileNotFoundError(f"No LAS/LAZ files in {path}")
    return result


def process_file(
    source: Path,
    destination: Path,
    *,
    k: int,
    reference_grid_size: float,
    chunk_size: int,
    overwrite: bool,
    backend: str = "cpu",
    device: str = "cuda:0",
) -> dict:
    if backend not in {"cpu", "gpu"}:
        raise ValueError("process_file backend must be 'cpu' or 'gpu'")
    if destination.exists() and not overwrite:
        return {"file": str(source), "status": "skipped_existing"}
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output LAS must differ; in-place mutation is disabled")
    start_time = time.perf_counter()
    las = laspy.read(source)
    names = set(las.point_format.dimension_names)
    required = (*NORMAL_NAMES, *DESCRIPTOR_NAMES, VALID_NAME)
    collision = names.intersection(required)
    if collision:
        raise ValueError(f"{source}: existing target dimensions {sorted(collision)}; refusing to overwrite")
    xyz = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))
    read_seconds = time.perf_counter() - start_time
    geometry_function = (
        compute_local_geometry_gpu if backend == "gpu" else compute_local_geometry
    )
    normal, descriptor, valid = geometry_function(
        xyz, k=k, reference_grid_size=reference_grid_size, chunk_size=chunk_size,
        **({"device": device} if backend == "gpu" else {}),
    )
    geometry_seconds = time.perf_counter() - start_time - read_seconds
    for name in (*NORMAL_NAMES, *DESCRIPTOR_NAMES):
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name=VALID_NAME, type=np.uint8))
    for index, name in enumerate(NORMAL_NAMES):
        las[name] = normal[:, index]
    for index, name in enumerate(DESCRIPTOR_NAMES):
        las[name] = descriptor[:, index]
    las[VALID_NAME] = valid.astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}{destination.suffix}")
    try:
        las.write(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"file": str(source), "status": "written", "points": len(las.points),
            "valid_rate": float(valid.mean()) if valid.size else 0.0,
            "read_seconds": read_seconds, "geometry_seconds": geometry_seconds,
            "write_seconds": time.perf_counter() - start_time - read_seconds - geometry_seconds}


def compute_las_local_geometry(
    input_path: str | Path,
    output_path: str | Path,
    *,
    recursive: bool = False,
    num_workers: int = 2,
    k: int = 48,
    reference_grid_size: float = 0.25,
    chunk_size: int = 8192,
    overwrite: bool = False,
    max_files: int | None = None,
    backend: str = "cpu",
    device: str = "cuda:0",
) -> list[dict]:
    source_root, output_root = Path(input_path), Path(output_path)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("Use a separate output directory; input LAS files must remain intact")
    files = _input_files(source_root, recursive)
    if max_files is not None:
        files = files[:max_files]
    if not files or num_workers < 1:
        raise ValueError("No files selected or num_workers < 1")
    backend = backend.lower()
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError("backend must be 'cpu', 'gpu' or 'auto'")
    if backend in {"gpu", "auto"}:
        import torch
        if backend == "auto":
            backend = "gpu" if torch.cuda.is_available() else "cpu"
        elif not torch.cuda.is_available():
            raise RuntimeError("GPU geometry backend requested but CUDA is unavailable")
    if backend == "gpu" and num_workers > 2:
        LOGGER.warning("GPU backend caps workers at 2; overriding num_workers=%d", num_workers)
        num_workers = 2
    LOGGER.info("Processing %d LAS/LAZ files, backend=%s, workers=%d, K=%d, reference grid=%.3f m",
                len(files), backend, num_workers, k, reference_grid_size)
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending = {}
        remaining = iter(files)

        def submit_next():
            try:
                path = next(remaining)
            except StopIteration:
                return
            future = pool.submit(
                process_file, path,
                output_root / (path.relative_to(source_root) if source_root.is_dir() else path.name),
                k=k, reference_grid_size=reference_grid_size,
                chunk_size=chunk_size, overwrite=overwrite,
                backend=backend, device=device,
            )
            pending[future] = path

        for _ in range(min(len(files), 2 * num_workers)):
            submit_next()
        with tqdm(total=len(files), desc="LAS geometry", unit="file") as progress:
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    source = pending.pop(future)
                    try:
                        results.append(future.result())
                    except Exception:
                        LOGGER.exception("Failed processing %s", source)
                        for unfinished in pending:
                            unfinished.cancel()
                        raise
                    progress.update(1)
                    submit_next()
    LOGGER.info("Finished: written=%d, skipped=%d",
                sum(r["status"] == "written" for r in results),
                sum(r["status"] == "skipped_existing" for r in results))
    written = [r for r in results if r["status"] == "written"]
    if written:
        LOGGER.info("Mean per file: read %.2fs, geometry %.2fs, write %.2fs",
                    *(sum(r[key] for r in written) / len(written)
                      for key in ("read_seconds", "geometry_seconds", "write_seconds")))
    return results


if __name__ == "__main__":
    compute_las_local_geometry(
        input_path=r"G:\data\swiss\joint_tiles_100\pointcloud",
        output_path=r"G:\data\swiss\joint_tiles_100\pointcloud_geom_p1",
        recursive=False,
        num_workers=8,
        k=8,
        reference_grid_size=0.25,
        chunk_size=8192,
        overwrite=False,
        max_files=None,
        backend="gpu",
        device="cuda:0",
    )
