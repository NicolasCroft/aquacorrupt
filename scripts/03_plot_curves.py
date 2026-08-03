"""
Step 3: plot the robustness curve from results/metrics.json (CPU, instant).

    python scripts/03_plot_curves.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.plotting import plot_curves


def main():
    metrics = json.loads((config.RESULTS_DIR / "metrics.json").read_text())
    # JSON turns int severity keys into strings; restore ints for plotting.
    for r in metrics["results"]:
        r["acc"] = {int(k): v for k, v in r["acc"].items()}
        r["retention"] = {int(k): v for k, v in r["retention"].items()}
    out = plot_curves(metrics["results"], metrics["corruption"],
                      config.RESULTS_DIR / "robustness_curve.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
