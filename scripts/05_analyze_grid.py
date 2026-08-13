"""
Step 5: k-fold CV probe over the full grid, with uncertainty.

Three things this does that 02_run_probe.py does not:

  1. K-FOLD CV. Every image gets a held-out prediction, so accuracy is estimated on all N
     instead of a single 185-image split. CIs shrink ~sqrt(k).
  2. CONFIDENCE INTERVALS. Paired bootstrap over images (the same resample indices are
     applied to every backbone, so differences are paired and the CI on a difference is
     much tighter than the CIs on the two endpoints suggest).
  3. ABSOLUTE ACCURACY ALONGSIDE RETENTION. mean_retention divides each backbone by its
     OWN clean accuracy, so a weak-but-flat backbone can outrank a strong one. Both
     numbers are reported; read them together.

    python scripts/05_analyze_grid.py

Writes results/grid_metrics.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import cache
from src.probe import fit_probe
from src.corruptions import CORRUPTIONS, FULL_SEVERITIES

NBOOT = 10000


def heldout_correct(emb_dir, bb, conditions, n_splits, seed):
    """Per-image held-out correctness for each condition.

    The probe is always FIT on CLEAN features of the training folds (ImageNet-C protocol)
    and EVALUATED on the held-out fold's features for the given condition. Returns
    {condition: bool array [N]} plus labels.
    """
    Xc, y = cache.load(emb_dir, bb, "all", "clean", 0)
    Xs = {}
    for c, s in conditions:
        Xi, yi = cache.load(emb_dir, bb, "all", c, s)
        if not np.array_equal(yi, y):
            raise ValueError(f"{bb} {c} s{s}: label order differs from clean cache")
        Xs[(c, s)] = Xi

    out = {k: np.zeros(len(y), dtype=bool) for k in Xs}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xc, y):
        scaler, clf = fit_probe(Xc[tr], y[tr], seed=seed)
        for k, Xi in Xs.items():
            out[k][te] = clf.predict(scaler.transform(Xi[te])) == y[te]
    return out, y


def main(backbones, corruptions, severities, n_splits):
    conditions = [("clean", 0)] + [(c, s) for c in corruptions for s in severities]
    missing = [(bb, c, s) for bb in backbones for c, s in conditions
               if not cache.exists(config.EMB_DIR, bb, "all", c, s)]
    if missing:
        print(f"[abort] {len(missing)} cached conditions missing, e.g. {missing[:3]}.")
        print("Run scripts/04_extract_full_grid.py first.")
        return None

    correct, labels = {}, None
    for bb in backbones:
        correct[bb], labels = heldout_correct(
            config.EMB_DIR, bb, conditions, n_splits, config.SEED)

    N = len(labels)
    maj = float(np.bincount(labels).max() / N)
    rng = np.random.default_rng(config.SEED)
    idx = rng.integers(0, N, size=(NBOOT, N))   # shared -> paired across backbones

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return float(lo), float(hi)

    print(f"N = {N} held-out predictions ({n_splits}-fold CV), "
          f"majority-class baseline = {maj:.3f}\n")

    results, boot = [], {}
    for bb in backbones:
        c = correct[bb]
        clean = c[("clean", 0)]
        clean_boot = clean[idx].mean(1)
        clean_acc = float(clean.mean())
        lo, hi = ci(clean_boot)
        entry = {"backbone": bb, "clean_acc": clean_acc,
                 "clean_acc_ci95": [lo, hi], "by_corruption": {}}
        boot[bb] = {}
        for corr in corruptions:
            accs = {s: float(c[(corr, s)].mean()) for s in severities}
            rets = {s: accs[s] / clean_acc for s in severities}
            acc_ci = {s: list(ci(c[(corr, s)][idx].mean(1))) for s in severities}
            mr_boot = np.mean(
                [c[(corr, s)][idx].mean(1) / np.maximum(clean_boot, 1e-9)
                 for s in severities], axis=0)
            ma_boot = np.mean([c[(corr, s)][idx].mean(1) for s in severities], axis=0)
            boot[bb][corr] = {"ret": mr_boot, "acc": ma_boot}
            mlo, mhi = ci(mr_boot)
            entry["by_corruption"][corr] = {
                "acc": accs, "acc_ci95": acc_ci, "retention": rets,
                "mean_retention": float(np.mean(list(rets.values()))),
                "mean_retention_ci95": [mlo, mhi],
                "mean_corrupted_acc": float(np.mean(list(accs.values()))),
            }
        results.append(entry)

    # Collect every pairwise test first: with 3 corruptions x 3 pairs = 9 tests at
    # alpha=0.05 you expect ~0.45 false "separations" by chance, so a raw CI excluding
    # zero is not on its own evidence. Holm-Bonferroni controls the family-wise error
    # rate across the whole grid and is uniformly more powerful than plain Bonferroni.
    tests = []
    for corr in corruptions:
        for a, b in itertools.combinations(backbones, 2):
            d = boot[a][corr]["ret"] - boot[b][corr]["ret"]
            lo, hi = ci(d)
            p = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
            tests.append({"corruption": corr, "a": a, "b": b, "diff": float(d.mean()),
                          "ci95": [lo, hi], "p_raw": p})
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i]["p_raw"])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, tests[i]["p_raw"] * (m - rank))
        running = max(running, adj)          # enforce monotonicity
        tests[i]["p_holm"] = running
        tests[i]["significant_holm_05"] = bool(running < 0.05)
    by_pair = {(t["corruption"], t["a"], t["b"]): t for t in tests}

    for corr in corruptions:
        print(f"--- {corr} ---")
        print(f'{"backbone":12s} {"clean":>7s}  ' +
              '  '.join(f"s{s}" for s in severities) +
              f'  {"mean acc":>9s}  {"mean ret":>9s}  {"ret 95% CI":>18s}')
        for e in results:
            b = e["by_corruption"][corr]
            lo, hi = b["mean_retention_ci95"]
            print(f'{e["backbone"]:12s} {e["clean_acc"]:.3f}  ' +
                  '  '.join(f'{b["acc"][s]:.3f}' for s in severities) +
                  f'  {b["mean_corrupted_acc"]:9.3f}  {b["mean_retention"]:9.3f}'
                  f'  [{lo:.3f},{hi:.3f}]')
        print("  pairwise mean-retention differences (paired bootstrap):")
        sep = False
        for a, b in itertools.combinations(backbones, 2):
            t = by_pair[(corr, a, b)]
            lo, hi = t["ci95"]
            survives = t["significant_holm_05"]
            sep = sep or survives
            note = ("   <-- survives Holm" if survives
                    else "   (raw CI excludes 0, NOT after Holm)"
                    if (lo > 0 or hi < 0) else "")
            print(f'    {a:11s} - {b:11s}: {t["diff"]:+.4f} [{lo:+.4f},{hi:+.4f}] '
                  f' p_raw~{t["p_raw"]:.3f}  p_holm~{t["p_holm"]:.3f}{note}')
        print(f"  => {'SEPARATION' if sep else 'no separation'} on {corr} "
              f"(family-wise corrected)\n")

    out = config.RESULTS_DIR / "grid_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": N, "n_splits": n_splits, "majority_baseline": maj,
        "n_bootstrap": NBOOT, "corruptions": corruptions,
        "severities": severities, "results": results,
        "pairwise_tests": tests, "multiple_comparison": "holm-bonferroni"}, indent=2))
    print(f"wrote {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=config.MVP_BACKBONES)
    ap.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS))
    ap.add_argument("--severities", nargs="*", type=int, default=FULL_SEVERITIES)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    main(args.backbones, args.corruptions, args.severities, args.folds)
