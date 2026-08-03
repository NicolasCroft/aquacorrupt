"""
Step 2: linear probe + robustness metrics (CPU, seconds).

    python scripts/02_run_probe.py

Writes results/metrics.json with clean_acc, per-severity acc, and retention per backbone.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.probe import evaluate_backbone


def main(backbones, corruption, severities):
    results = []
    for bb in backbones:
        try:
            r = evaluate_backbone(config.EMB_DIR, bb, corruption, severities, seed=config.SEED)
        except FileNotFoundError as e:
            print(f"[skip] {bb}: missing features ({e}). Run 01_extract_features first.")
            continue
        results.append(r)
        print(f"{bb:12s} clean={r['clean_acc']:.3f}  mean_retention={r['mean_retention']:.3f}"
              f"  acc={ {s: round(a,3) for s,a in r['acc'].items()} }")

    out = config.RESULTS_DIR / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"corruption": corruption, "severities": severities, "results": results}, indent=2))
    print(f"\nwrote {out}")
    return results


if __name__ == "__main__":
    main(config.MVP_BACKBONES, config.CORRUPTION, config.SEVERITIES)
