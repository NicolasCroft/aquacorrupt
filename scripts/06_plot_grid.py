"""
Step 6: the full-grid figure -- small multiples, one panel per corruption.

Reads results/grid_metrics.json (written by 05_analyze_grid.py) and draws held-out
accuracy vs severity, one line per backbone, with bootstrap 95% confidence bands.

Design notes (why it looks like this):
  - Small multiples with a SHARED y-axis, never a second y-scale. The three corruptions
    have different strengths; a shared axis is what makes them comparable at a glance.
  - Confidence bands are the point of the figure. The MVP's flat curves looked like a
    ranking until you saw the intervals overlap; bands make that impossible to misread.
  - Direct labels on every series in the last panel, in addition to the legend, so
    identity never depends on color alone (one series color is below 3:1 on this
    surface). grid_metrics.json is the table view.
  - The majority-class baseline is drawn as a recessive dashed rule: accuracy above it
    is the only accuracy that means anything.

    python scripts/06_plot_grid.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# Categorical slots 1-3 of the validated palette. Fixed order, assigned per backbone
# (color follows the entity, never its rank), so a re-ordered legend never repaints.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#87867f"
GRID = "#e3e2dd"


def plot_grid(metrics, out_path, title=None):
    corruptions = metrics["corruptions"]
    severities = metrics["severities"]
    results = metrics["results"]
    baseline = metrics["majority_baseline"]

    # Fixed entity -> color map, independent of plotting order.
    order = [r["backbone"] for r in results]
    cmap = {bb: SERIES_COLORS[i % len(SERIES_COLORS)] for i, bb in enumerate(order)}

    # Wrap to at most 3 panels per row: 6 side by side would be ~25in wide.
    ncols = min(3, len(corruptions))
    nrows = -(-len(corruptions) // ncols)
    fig, axgrid = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.0 * nrows),
                               dpi=150, sharey=True, squeeze=False)
    axes = axgrid.ravel()
    for extra in axes[len(corruptions):]:      # hide unused cells
        extra.set_visible(False)
    axes = list(axes[:len(corruptions)])

    for ax, corr in zip(axes, corruptions):
        ax.axhline(baseline, color=TEXT_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for r in results:
            bb = r["backbone"]
            b = r["by_corruption"][corr]
            xs = [0] + severities
            ys = [r["clean_acc"]] + [b["acc"][str(s)] for s in severities]
            lo = [r["clean_acc_ci95"][0]] + [b["acc_ci95"][str(s)][0] for s in severities]
            hi = [r["clean_acc_ci95"][1]] + [b["acc_ci95"][str(s)][1] for s in severities]
            ax.fill_between(xs, lo, hi, color=cmap[bb], alpha=0.13, lw=0, zorder=2)
            ax.plot(xs, ys, color=cmap[bb], lw=2.0, marker="o", ms=5.5,
                    mec="white", mew=1.2, zorder=3, label=bb, clip_on=False)

        ax.set_title(corr, fontsize=11, color=TEXT_PRIMARY, pad=8)
        ax.set_xlabel("severity  (0 = clean)", fontsize=9.5, color=TEXT_SECONDARY)
        ax.set_xticks([0] + severities)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

    for row in range(nrows):                   # y-label on the first panel of each row
        i = row * ncols
        if i < len(axes):
            axes[i].set_ylabel("held-out linear-probe accuracy", fontsize=9.5,
                               color=TEXT_SECONDARY)
    # Adaptive limits: the augmentation probes can drop accuracy far below the
    # physical corruptions' floor, and a hardcoded ylim would clip them off-panel.
    lows = [b["acc"][str(s)] for r in results for b in r["by_corruption"].values()
            for s in severities]
    highs = [r["clean_acc"] for r in results] + lows
    axes[0].set_ylim(min(min(lows), baseline) - 0.06, max(highs) + 0.05)
    axes[0].annotate("majority-class baseline", xy=(0.04, baseline + 0.012),
                     xycoords=("axes fraction", "data"), fontsize=8, color=TEXT_MUTED)

    # Relief rule: direct labels so identity is never color-alone. Series can finish
    # within a hair of each other (dinov2/supervised differ by ~0.01 under attenuation),
    # so nudge labels apart to a minimum gap instead of letting them overprint.
    last, xmax = axes[-1], max(severities)
    ends = sorted(((r["by_corruption"][corruptions[-1]]["acc"][str(xmax)], r["backbone"])
                   for r in results), reverse=True)
    lo_lim, hi_lim = axes[0].get_ylim()
    min_gap = 0.030 * (hi_lim - lo_lim)
    placed = []
    for y, bb in ends:
        if placed and placed[-1][0] - y < min_gap:
            y = placed[-1][0] - min_gap
        placed.append((y, bb))
    for (y_text, bb), (y_true, _) in zip(placed, ends):
        last.annotate(bb, xy=(xmax, y_true), xytext=(xmax + 0.18, y_text),
                      textcoords="data", va="center", fontsize=9,
                      color=TEXT_SECONDARY, annotation_clip=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(results), frameon=False,
               fontsize=9.5, labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        (title or "Frozen-feature robustness to air-water corruption") +
        f' ({metrics["n"]} held-out predictions, {metrics["n_splits"]}-fold CV, '
        f'bands = 95% bootstrap CI)',
        fontsize=11.5, color=TEXT_PRIMARY, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 0.985, 0.96))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(config.RESULTS_DIR / "grid_metrics.json"))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "robustness_grid.png"))
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    src = Path(args.metrics)
    if not src.exists():
        print(f"[abort] {src} missing. Run scripts/05_analyze_grid.py first.")
        sys.exit(1)
    p = plot_grid(json.loads(src.read_text()), args.out, args.title)
    print(f"wrote {p}")
