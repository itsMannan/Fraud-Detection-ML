"""Feature engineering fitted on the training split only (no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.config import NIGHT_HOURS


class FeatureEngineer:
    """Add temporal + amount features using statistics learned from train."""

    def __init__(self) -> None:
        self.amount_mean_: float = 0.0
        self.amount_std_: float = 1.0
        self.amount_median_: float = 0.0
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        if "Amount" in df.columns:
            amounts = df["Amount"].astype(float)
            self.amount_mean_ = float(amounts.mean())
            std = float(amounts.std(ddof=0))
            self.amount_std_ = std if std > 1e-9 else 1.0
            self.amount_median_ = float(amounts.median())
        self._fitted = True
        # Dry-run to lock column order.
        self.feature_names_ = [
            c for c in self.transform(df).columns if c != "Class"
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.fit() must be called before transform().")
        out = df.copy()
        if "Amount" in out.columns:
            amount = out["Amount"].astype(float).clip(lower=0)
            out["Amount_log"] = np.log1p(amount)
            out["Amount_to_mean"] = amount / (self.amount_mean_ + 1e-9)
            out["Amount_zscore"] = (amount - self.amount_mean_) / self.amount_std_
        if "Time" in out.columns:
            hours = (out["Time"].astype(float) % 86_400) / 3_600.0
            out["Hour"] = hours
            out["Hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            out["Hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
            lo, hi = NIGHT_HOURS
            out["Is_night"] = ((hours >= lo) & (hours < hi)).astype(int)
        if "Class" in out.columns:
            features = [c for c in out.columns if c != "Class"]
            return out[features + ["Class"]]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "Class" not in df.columns:
        raise ValueError("Expected a Class column")
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y
