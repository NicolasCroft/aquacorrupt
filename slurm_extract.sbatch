#!/bin/bash
#SBATCH --job-name=aquacorrupt
#SBATCH --partition=gpu            # change to your cluster's GPU partition (e.g. gpu, owners)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=aquacorrupt_%j.out

# Feature extraction (+ probe + plot) on a SLURM cluster (e.g. Stanford Sherlock).
# Submit with:   sbatch slurm_extract.sbatch
# Everything downstream of features is CPU-cheap, so we just run it all in the job.

set -e
echo "host: $(hostname)  |  job: $SLURM_JOB_ID"
nvidia-smi -L || true

# --- environment: adjust to your cluster --------------------------------------
# module load python/3.11 cuda/12.4      # if your cluster uses Lmod modules
# source ~/envs/aquacorrupt/bin/activate # or your conda/venv
# On first run only: pip install -r requirements.txt

# --- data ---------------------------------------------------------------------
# The dataset must already be staged (Kaggle download needs auth, so do it on a
# login node before submitting):
#   python scripts/00_download_data.py --download
# then confirm config.TRAIN_DIR / TEST_DIR point at the class folders it reports.

# --- run ----------------------------------------------------------------------
python scripts/01_extract_features.py --device cuda
python scripts/02_run_probe.py
python scripts/03_plot_curves.py

echo "done. results/robustness_curve.png and results/metrics.json written."
