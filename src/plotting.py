"""Robustness curves: accuracy vs corruption severity, one line per backbone.
The headline plot of the whole MVP. If the lines separate, the pretraining objective
matters for air-water robustness and the full study is worth building."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(results, corruption, out_path):
    """results: list of dicts from probe.evaluate_backbone."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=130)

    for r in sorted(results, key=lambda d: -d["mean_retention"]):
        sev = [0] + sorted(r["acc"])
        accs = [r["clean_acc"]] + [r["acc"][s] for s in sorted(r["acc"])]
        ax.plot(sev, accs, marker="o", linewidth=2,
                label=f'{r["backbone"]}  (ret={r["mean_retention"]:.2f})')

    ax.set_xlabel(f"{corruption} severity (0 = clean)")
    ax.set_ylabel("linear-probe accuracy")
    ax.set_title(f"Representation robustness to {corruption}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(title="backbone (mean retention)", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
