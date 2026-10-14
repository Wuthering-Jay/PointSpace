"""Generate the five DALES Scratch FT label-collapse probe configs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.create_sparse_label_manifests import COLLAPSE_FRACTIONS
from utils.generate_downstream_ssl_configs import render


def generate(output_dir="configs/label_efficiency/dales/scratch_collapse"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for fraction in COLLAPSE_FRACTIONS:
        tag = f"{fraction * 100:g}pct".replace(".", "p")
        path = output / f"semseg-scratch-litept-v1m4-dales-ft-{tag}.py"
        path.write_text(
            render(
                "dales", "scratch", "ft", label_fraction=fraction,
                save_path_root="exp/label_efficiency/dales/scratch_collapse",
            ),
            encoding="utf-8",
        )
    return len(COLLAPSE_FRACTIONS)


if __name__ == "__main__":
    print(f"Generated {generate()} Scratch collapse configs")
