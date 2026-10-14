"""Generate standalone Section 4.6 DALES sparse-label FT configurations."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.generate_downstream_ssl_configs import METHODS, render

MAIN_CURVE_FRACTIONS = (0.0001, 0.001, 0.01, 0.1)


def generate(output_dir="configs/label_efficiency/dales"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for fraction in MAIN_CURVE_FRACTIONS:
        tag = f"{fraction * 100:g}pct".replace(".", "p")
        for method in METHODS:
            path = output / f"semseg-{method}-litept-v1m4-dales-ft-{tag}.py"
            path.write_text(render("dales", method, "ft", label_fraction=fraction), encoding="utf-8")
    return len(MAIN_CURVE_FRACTIONS) * len(METHODS)


if __name__ == "__main__":
    print(f"Generated {generate()} sparse-label DALES FT configs")
