"""
Competition-specific configuration.
"""

from pathlib import Path

# Paths
COMPETITION_ROOT = Path(__file__).parent
DATA_RAW = COMPETITION_ROOT / "data" / "raw"
DATA_PROCESSED = COMPETITION_ROOT / "data" / "processed"
MODELS_DIR = COMPETITION_ROOT / "models"
SUBMISSIONS_DIR = COMPETITION_ROOT / "submissions"

# Competition settings
COMPETITION_SLUG = "competition-name"  # Update this
TARGET_COL = "target"  # Update this
ID_COL = "id"  # Update this

# Task type
TASK = "classification"  # or "regression"

# Cross-validation
N_FOLDS = 5
SEED = 42

# Training
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 1000

# Features to exclude
EXCLUDE_COLS = [TARGET_COL, "fold", ID_COL]
