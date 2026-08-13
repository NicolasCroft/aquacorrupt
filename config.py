"""Central config. Everything you'd want to change for the MVP lives here."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"          # you place datasets here (see scripts/00_download_data.py)
EMB_DIR = ROOT / "embeddings"     # cached feature tensors (.npz)
RESULTS_DIR = ROOT / "results"    # metrics + figures

SEED = 0
IMG_SIZE = 224
BATCH_SIZE = 64
MAX_PER_CLASS = 1000              # cap images/class to keep the MVP an afternoon, not a week
NUM_WORKERS = 4

# --- Backbones ---------------------------------------------------------------
# The MVP trio: three DIFFERENT pretraining objectives, each loadable in one line.
#   mae        -> reconstructive SSL (the thing JEPA argues against)
#   dinov2     -> non-reconstructive joint-embedding SSL (JEPA-adjacent, 1-line load)
#   supervised -> label-trained control
# For the paper, add true I-JEPA (backbones.load_ijepa) and an EO arm (AnySat/Core-JEPA).
MVP_BACKBONES = ["mae", "dinov2", "supervised"]

# --- Corruption grid ---------------------------------------------------------
# MVP: glint only at 3 severities. Full grid lives in src/corruptions.py.
from src.corruptions import MVP_CORRUPTION, MVP_SEVERITIES  # noqa: E402
CORRUPTION = MVP_CORRUPTION
SEVERITIES = MVP_SEVERITIES

# --- Dataset -----------------------------------------------------------------
# Default MVP dataset: Kaggle "Healthy and Bleached Corals" (simple ImageFolder).
#
# OBSERVED LAYOUT (as downloaded 2026-08-13 via scripts/00_download_data.py --download):
# the zip extracts FLAT, with no train/valid split --
#     data/corals/bleached_corals/*.jpg   (485)
#     data/corals/healthy_corals/*.jpg    (438)
# so the two paths below do not exist. That is intentional: src.data.prepare_datasets
# falls through to its single-folder case and makes a deterministic seeded 80/20
# stratified split (738 train / 185 test).
#
# Do NOT "fix" this by pointing both at data/corals -- prepare_datasets would then find
# class dirs at both and train and test on the same 923 images, leaking the test set.
# Only set these to real, disjoint folders if you obtain a pre-split copy.
TRAIN_DIR = DATA_DIR / "corals" / "train"
TEST_DIR = DATA_DIR / "corals" / "valid"
