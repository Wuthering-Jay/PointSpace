"""Create nested, compact DALES label masks without copying point clouds.

Edit the constants at the bottom when running directly in an IDE. The source
LAS files are read, never modified. Each NPZ stores only the points admitted
by the largest budget and their common random ranks.
"""

import hashlib
import json
from pathlib import Path

import laspy
import numpy as np
from tqdm import tqdm


FRACTIONS = (0.001, 0.01, 0.05, 0.10)
COLLAPSE_FRACTIONS = (0.0001, 0.0003, 0.001, 0.003, 0.01)
MANIFEST_FRACTIONS = tuple(sorted(set(FRACTIONS + COLLAPSE_FRACTIONS)))
DALES_CLASSES = tuple(range(1, 9))


def _file_seed(seed, name):
    digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def create_manifests(input_dir, output_dir, *, seed=42, fractions=MANIFEST_FRACTIONS):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    fractions = tuple(sorted(float(x) for x in fractions))
    if not fractions or fractions[0] <= 0 or fractions[-1] >= 1:
        raise ValueError("fractions must lie strictly between 0 and 1")
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".las", ".laz"})
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files in {input_dir}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("LAS/LAZ files with duplicate stems would overwrite manifests")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    global_counts = {str(f): np.zeros(len(DALES_CLASSES), np.int64) for f in fractions}
    source_class_counts = np.zeros(len(DALES_CLASSES), np.int64)
    total_points = 0
    for path in tqdm(files, desc="Sparse label manifests", unit="tile"):
        las = laspy.read(path)
        labels = np.asarray(las.classification)
        n = len(labels)
        full_counts = np.bincount(labels, minlength=max(DALES_CLASSES) + 1)[
            list(DALES_CLASSES)
        ].astype(np.int64)
        source_class_counts += full_counts
        # A single per-file RNG stream makes every selection exactly nested.
        rank = np.random.default_rng(_file_seed(seed, path.name)).random(n, dtype=np.float32)
        selected = np.flatnonzero(rank < fractions[-1]).astype(np.uint32)
        selected_rank = rank[selected]
        np.savez_compressed(
            output_dir / f"{path.stem}.npz",
            index=selected,
            rank=selected_rank,
            point_count=np.array(n, dtype=np.int64),
        )
        counts = {}
        for fraction in fractions:
            chosen = selected[selected_rank < fraction]
            class_counts = np.bincount(labels[chosen], minlength=max(DALES_CLASSES) + 1)[
                list(DALES_CLASSES)
            ].astype(np.int64)
            counts[str(fraction)] = class_counts.tolist()
            global_counts[str(fraction)] += class_counts
        records[path.name] = {
            "point_count": n,
            "source_bytes": path.stat().st_size,
            "source_mtime_ns": path.stat().st_mtime_ns,
            "source_class_counts": full_counts.tolist(),
            "counts": counts,
        }
        total_points += n
    summary = {
        "format": "pointspace-sparse-label-v1",
        "seed": seed,
        "fractions": fractions,
        "classes": DALES_CLASSES,
        "source_dir": str(input_dir.resolve()),
        "total_points": total_points,
        "source_class_counts": source_class_counts.tolist(),
        "global_counts": {key: value.tolist() for key, value in global_counts.items()},
        "files": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    INPUT_DIR = r"E:\data\DALES\dales_las\tile\train"
    OUTPUT_DIR = r"E:\data\DALES\dales_las\tile\sparse_labels_seed42"
    create_manifests(INPUT_DIR, OUTPUT_DIR)
