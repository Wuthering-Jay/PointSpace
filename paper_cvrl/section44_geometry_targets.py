"""Prototype geometry-only targets for the Section 4.4 P1 controlled baseline.

Compute on the *current* cropped/grid-sampled XYZ, independently per sample.
The returned normal is an optional diagnostic; the primary targets are local
surface variation (curvature) and planarity. No LAS labels, return attributes,
visibility score, or image features are read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GeometryTargets:
    curvature: np.ndarray
    planarity: np.ndarray
    normal: np.ndarray
    valid: np.ndarray
    neighbor_radius: np.ndarray


def local_geometry_targets(
    xyz: np.ndarray,
    *,
    k: int = 32,
    chunk_size: int = 16384,
    min_neighbors: int = 8,
    min_variance: float = 1e-10,
) -> GeometryTargets:
    """Estimate covariance-eigenvalue geometry targets with exact XYZ KNN.

    The center itself is explicitly removed even when coincident points make
    nearest-neighbor ordering ambiguous. A sample with fewer than
    ``min_neighbors`` other points is invalid. Targets are bounded in [0, 1].
    ``neighbor_radius`` is the farthest chosen neighbor distance and can be
    used for a later physical-scale sensitivity check.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be a finite [N, 3] array")
    if k < 3 or chunk_size < 1 or min_neighbors < 3:
        raise ValueError("k, chunk_size, and min_neighbors must be positive and meaningful")
    n = xyz.shape[0]
    curvature = np.zeros(n, np.float32)
    planarity = np.zeros(n, np.float32)
    normal = np.zeros((n, 3), np.float32)
    radius = np.zeros(n, np.float32)
    valid = np.zeros(n, bool)
    count = min(k, n - 1)
    if count < min_neighbors:
        return GeometryTargets(curvature, planarity, normal, valid, radius)
    tree = cKDTree(xyz)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        centers = np.arange(start, stop)
        # Query one extra because self is among the K+1 nearest even with ties.
        _, indices = tree.query(xyz[start:stop], k=count + 1, workers=1)
        indices = np.atleast_2d(indices)
        keep = indices != centers[:, None]
        # Every row must contain at least count non-self neighbors. cKDTree may
        # return coincident duplicates in arbitrary order, hence explicit mask.
        if np.any(keep.sum(axis=1) < count):
            raise RuntimeError("KNN query returned too few non-self points")
        rank = np.argsort(~keep, axis=1, kind="stable")[:, :count]
        neighbors = np.take_along_axis(indices, rank, axis=1)
        coordinates = xyz[neighbors]
        centered = coordinates - coordinates.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / count
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        lam3, lam2, lam1 = (eigenvalues[:, i] for i in (0, 1, 2))
        denominator = np.maximum(eigenvalues.sum(axis=1), min_variance)
        is_valid = (lam1 > min_variance) & (np.isfinite(eigenvalues).all(axis=1))
        valid[start:stop] = is_valid
        curvature[start:stop] = np.where(is_valid, lam3 / denominator, 0).astype(np.float32)
        planarity[start:stop] = np.where(
            is_valid, (lam2 - lam3) / np.maximum(lam1, min_variance), 0
        ).astype(np.float32)
        normal[start:stop] = eigenvectors[:, :, 0].astype(np.float32)
        radius[start:stop] = np.linalg.norm(coordinates - xyz[centers, None], axis=2).max(axis=1)
    return GeometryTargets(curvature, planarity, normal, valid, radius)
