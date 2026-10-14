"""Export every Swiss-Audit representation required by Section 4.3.

This file intentionally keeps its configuration at module level so it can be
run directly from an IDE.  Each task uses the normal PointSpace tester and the
same fixed ``audit_index`` sidecars.  Existing complete files are skipped by
the tester unless ``OVERWRITE`` is enabled.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from safetensors import safe_open
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
OVERWRITE = False

VPSD_CONFIG = PROJECT_ROOT / "configs/vpsd/pretrain-vpsd-v1m1-litept-v1m4-swiss-100.py"
CVRL_CONFIG = PROJECT_ROOT / "configs/cvrl/audit-cvrl-v1m1-litept-v1m4-swiss-100.py"
VPSD_SCRATCH_CHECKPOINT = (
    PROJECT_ROOT / "paper_cvrl/tmp/vpsd_scratch_swiss100_seed42.pth"
)

# The order keeps the compact/default paper representations first.  A failed
# task stops the sequence, so a partially exported source is never hidden by a
# later successful task.
EXPORTS = (
    ("VPSD-128", VPSD_CONFIG, "visual", "vpsd_pca128", ()),
    ("VPSD-1024", VPSD_CONFIG, "visual_native", "vpsd_native1024", ()),
    # The 128D visual and 72D return branches are exact channel slices of the
    # full 200D feature. Export full once and split it below to avoid running
    # the same CVRL backbone three times.
    ("CVRL-full-200", CVRL_CONFIG, "full", "cvrl_full", ()),
    (
        "CVRL-visual-1024",
        CVRL_CONFIG,
        "visual_native",
        "cvrl_visual_native1024",
        (),
    ),
    (
        "VPSD-Scratch-128",
        VPSD_CONFIG,
        "visual",
        "vpsd_scratch_pca128",
        (f"weight={VPSD_SCRATCH_CHECKPOINT}", "resume=False"),
    ),
)


def _count_outputs(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.safetensors"))


def _derive_cvrl_branches(expected: int) -> None:
    source_dir = AUDIT_ROOT / "features" / "cvrl_full"
    destinations = {
        "visual": (AUDIT_ROOT / "features" / "cvrl_visual_pca128", slice(0, 128)),
        "return": (AUDIT_ROOT / "features" / "rrcl", slice(128, 200)),
    }
    source_files = sorted(source_dir.glob("*.safetensors"))
    if len(source_files) != expected:
        raise RuntimeError(
            f"CVRL-full split requires {expected} files, found {len(source_files)}"
        )
    for output_dir, _ in destinations.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_files:
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            metadata = dict(handle.metadata() or {})
        feature = tensors["feature"]
        if feature.ndim != 2 or feature.shape[1] != 200:
            raise ValueError(f"{source_path} is not a CVRL full-200 feature")
        for feature_source, (output_dir, channel_slice) in destinations.items():
            output_path = output_dir / source_path.name
            if output_path.exists() and not OVERWRITE:
                continue
            branch = feature[:, channel_slice].contiguous()
            branch_tensors = {"feature": branch}
            if "point_index" in tensors:
                branch_tensors["point_index"] = tensors["point_index"].contiguous()
            branch_metadata = {
                key: value
                for key, value in metadata.items()
                if key not in {"visual_slice", "return_slice"}
            }
            branch_metadata.update(
                feature_source=feature_source,
                feature_layout=feature_source,
                feature_dim=str(branch.shape[1]),
                derived_from="cvrl_full200_exact_channel_slice",
            )
            temporary_path = output_path.with_name(
                f".{output_path.name}.{uuid4().hex}.tmp"
            )
            save_file(branch_tensors, str(temporary_path), branch_metadata)
            temporary_path.replace(output_path)
    for name, (output_dir, _) in destinations.items():
        count = _count_outputs(output_dir)
        if count != expected:
            raise RuntimeError(f"CVRL-{name} split incomplete: {count}/{expected}")
        print(f"[CVRL-{name}] ready={count} (exact full-200 slice)", flush=True)


def export_all() -> None:
    test_script = PROJECT_ROOT / "tools/test.py"
    expected = sum(1 for _ in (AUDIT_ROOT / "pointcloud").glob("*.la[sz]"))
    if expected == 0:
        raise FileNotFoundError(f"No Swiss-Audit LAS/LAZ files under {AUDIT_ROOT}")
    for name, config, source, subdirectory, extra_options in EXPORTS:
        output = AUDIT_ROOT / "features" / subdirectory
        output.mkdir(parents=True, exist_ok=True)
        before = _count_outputs(output)
        print(f"\n[{name}] existing={before}, output={output}", flush=True)
        if before == expected and not OVERWRITE:
            print(f"[{name}] complete ({before}/{expected}); skipped", flush=True)
            continue
        command = [
            str(PYTHON),
            str(test_script),
            "--config-file",
            str(config),
            "--options",
            f"feature_source={source}",
            f"feature_output_dir={output}",
            f"feature_overwrite={OVERWRITE}",
            *extra_options,
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        after = _count_outputs(output)
        if after != expected:
            raise RuntimeError(
                f"{name} export incomplete: expected {expected}, found {after}"
            )
        print(f"[{name}] ready={after}", flush=True)
    _derive_cvrl_branches(expected)


if __name__ == "__main__":
    export_all()
