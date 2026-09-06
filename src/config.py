"""
Configuration: paths and constants for the SHAP stability audit.

"""

from pathlib import Path

# Derived from this file's own location, so the project works
# wherever it is cloned to.
ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Reproducibility ---
RANDOM_SEED = 42
TEST_SIZE = 0.2

# --- Audit settings ---
STABILITY_THRESHOLD = 0.90   # Spearman rho at or above this = "stable"
TOP_K = 3                    # how many top features to check for overlap

# --- Default column names ---
# Defaults only. The audit functions take these as arguments so the
# tool works on datasets that name their columns differently.
DEFAULT_AGE_COL = "Age"
DEFAULT_GENDER_COL = "Gender"
DEFAULT_TARGET_COL = "HiringDecision"