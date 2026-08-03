#!/usr/bin/env bash
# End-to-end MVP driver. Assumes data is in place (see scripts/00_download_data.py).
set -e
DEVICE="${1:-cpu}"   # pass 'cuda' if you have a GPU: ./run_mvp.sh cuda
echo ">> extracting features on $DEVICE"
python scripts/01_extract_features.py --device "$DEVICE"
echo ">> running probe"
python scripts/02_run_probe.py
echo ">> plotting"
python scripts/03_plot_curves.py
echo ">> done. see results/robustness_curve.png and results/metrics.json"
