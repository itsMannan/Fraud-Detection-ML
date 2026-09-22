"""Project-wide constants and business thresholds."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15  # fraction of the training split, used for early stopping + threshold tuning
N_CV_FOLDS = 5
CV_SUBSAMPLE = 80_000  # stratified subsample so 5-fold CV stays tractable

# Public ULB credit-card fraud dataset mirrors (Kaggle mlg-ulb/creditcardfraud).
DATA_URLS = (
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
    "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
)

DEFAULT_DATA_PATH = ROOT / "data" / "creditcard.csv"
PLACEHOLDER_CSV = ROOT / "creditcard-selected-columns.csv"

ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"
EXAMPLES = ROOT / "examples"

# Business risk bands (simple score = P(fraud) * 100).
RISK_BINS = [
    (0, 30, "Low", "Approve immediately"),
    (30, 60, "Medium", "Manual review recommended"),
    (60, 80, "High", "Block and notify customer"),
    (80, 101, "Critical", "Block and escalate to fraud team"),
]

# Success criteria from the project spec.
TARGETS = {
    "precision": (0.75, 0.85),
    "recall": (0.80, 0.90),
    "f1": (0.75, 0.85),
    "pr_auc": (0.80, 1.0),
}

# Composite risk-score weights (no location feature in this dataset).
COMPOSITE_WEIGHTS = {
    "fraud_probability": 0.70,
    "amount_deviation": 0.20,
    "unusual_time": 0.10,
}

NIGHT_HOURS = (0.0, 6.0)  # [00:00, 06:00) treated as unusual
