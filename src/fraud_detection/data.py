"""Load, validate, and (if needed) download the credit-card fraud dataset."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from fraud_detection.config import DATA_URLS, DEFAULT_DATA_PATH, PLACEHOLDER_CSV, RANDOM_STATE


def _is_usable(path: Path) -> bool:
    """True when the CSV has rows, a Class label, and is not an empty placeholder."""
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    try:
        head = pd.read_csv(path, nrows=20)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if "Class" not in head.columns:
        return False
    return bool(head.dropna(how="all").shape[0])


def download_dataset(dest: Path, urls: tuple[str, ...] = DATA_URLS) -> Path:
    """Download the public ULB credit-card fraud CSV to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading dataset from {url}")
            urllib.request.urlretrieve(url, dest)
            if dest.exists() and dest.stat().st_size > 10_000_000:
                print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
                return dest
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
            print(f"  failed: {exc}")
    raise RuntimeError(f"Could not download the fraud dataset. Last error: {last_error}")


def resolve_data_path(explicit: str | Path | None = None) -> Path:
    """Pick the first usable CSV, downloading the public dataset if needed."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            DEFAULT_DATA_PATH,
            Path.cwd() / "data" / "creditcard.csv",
            Path.cwd() / "creditcard.csv",
            PLACEHOLDER_CSV,
        ]
    )
    for path in candidates:
        if _is_usable(path):
            return path
    return download_dataset(DEFAULT_DATA_PATH)


def load_transactions(path: str | Path | None = None) -> pd.DataFrame:
    """Load the transaction table, coerce types, and drop empty rows."""
    csv_path = resolve_data_path(path)
    print(f"Loading transactions from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    df = df.dropna(how="all")
    if "Class" not in df.columns:
        raise ValueError(
            f"{csv_path} has no 'Class' column. The repo placeholder CSV is empty; "
            "the pipeline will auto-download the public ULB dataset on the next run."
        )
    df["Class"] = df["Class"].astype(int)
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Class"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing Class")
    return df.reset_index(drop=True)


def describe_dataset(df: pd.DataFrame) -> dict:
    """Return headline stats used in logs and the HTML/Markdown report."""
    n = len(df)
    n_fraud = int((df["Class"] == 1).sum())
    n_legit = n - n_fraud
    ratio = (n_legit / n_fraud) if n_fraud else float("inf")
    stats = {
        "n_rows": n,
        "n_columns": int(df.shape[1]),
        "n_fraud": n_fraud,
        "n_legitimate": n_legit,
        "fraud_rate": n_fraud / n if n else 0.0,
        "imbalance_ratio": ratio,
        "missing_values": int(df.isna().sum().sum()),
        "amount_mean": float(df["Amount"].mean()) if "Amount" in df.columns else None,
        "amount_fraud_mean": float(df.loc[df["Class"] == 1, "Amount"].mean())
        if "Amount" in df.columns
        else None,
        "amount_legit_mean": float(df.loc[df["Class"] == 0, "Amount"].mean())
        if "Amount" in df.columns
        else None,
        "columns": list(df.columns),
    }
    print("=" * 70)
    print("STEP 1: DATA LOADING & EXPLORATION")
    print("=" * 70)
    print(f"  Transactions : {n:,}")
    print(f"  Features     : {df.shape[1]}")
    print(f"  Legitimate   : {n_legit:,} ({100 - stats['fraud_rate'] * 100:.3f}%)")
    print(f"  Fraudulent   : {n_fraud:,} ({stats['fraud_rate'] * 100:.3f}%)")
    print(f"  Imbalance    : 1:{ratio:.0f}")
    print(f"  Missing cells: {stats['missing_values']}")
    return stats


def make_synthetic_dataset(
    n_samples: int = 8_000,
    n_features: int = 12,
    fraud_rate: float = 0.02,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Build a tiny imbalanced table for tests and --demo runs."""
    from sklearn.datasets import make_classification

    n_fraud = max(int(n_samples * fraud_rate), 20)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(6, n_features // 2),
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[1 - fraud_rate, fraud_rate],
        class_sep=1.4,
        random_state=random_state,
    )
    columns = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=columns)
    rng = __import__("numpy").random.default_rng(random_state)
    df["Time"] = rng.uniform(0, 172_800, size=n_samples)
    df["Amount"] = __import__("numpy").abs(rng.lognormal(mean=3.2, sigma=1.1, size=n_samples))
    df["Class"] = y.astype(int)
    # Keep at least n_fraud positives in case make_classification undershot.
    if int(df["Class"].sum()) < n_fraud:
        df.loc[df.sample(n_fraud, random_state=random_state).index, "Class"] = 1
    return df
