"""Convert fraud probability into an actionable 0–100 risk score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.config import COMPOSITE_WEIGHTS, NIGHT_HOURS, RISK_BINS


def categorize_risk(score: float) -> tuple[str, str]:
    for lo, hi, label, action in RISK_BINS:
        if lo <= score < hi:
            return label, action
    return RISK_BINS[-1][2], RISK_BINS[-1][3]


def simple_risk_score(fraud_proba: np.ndarray) -> np.ndarray:
    """Risk_Score = P(fraud) × 100."""
    return np.clip(np.asarray(fraud_proba, dtype=float) * 100.0, 0.0, 100.0)


def amount_deviation_01(amount: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Map |amount - mean| into [0, 1] using a 3-sigma cap."""
    std = std if std and std > 1e-9 else 1.0
    z = np.abs(np.asarray(amount, dtype=float) - mean) / (3.0 * std)
    return np.clip(z, 0.0, 1.0)


def unusual_time_01(hour: np.ndarray) -> np.ndarray:
    lo, hi = NIGHT_HOURS
    hour = np.asarray(hour, dtype=float)
    return ((hour >= lo) & (hour < hi)).astype(float)


def composite_risk_score(
    fraud_proba: np.ndarray,
    amount: np.ndarray,
    hour: np.ndarray,
    amount_mean: float,
    amount_std: float,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted blend of model probability, amount unusualness, and night flag."""
    w = weights or COMPOSITE_WEIGHTS
    p = np.asarray(fraud_proba, dtype=float)
    amt = amount_deviation_01(amount, amount_mean, amount_std)
    night = unusual_time_01(hour)
    blended = (
        w["fraud_probability"] * p
        + w["amount_deviation"] * amt
        + w["unusual_time"] * night
    )
    return np.clip(blended * 100.0, 0.0, 100.0)


def score_frame(
    fraud_proba: np.ndarray,
    amount: np.ndarray | None = None,
    hour: np.ndarray | None = None,
    amount_mean: float = 0.0,
    amount_std: float = 1.0,
) -> pd.DataFrame:
    simple = simple_risk_score(fraud_proba)
    if amount is not None and hour is not None:
        composite = composite_risk_score(
            fraud_proba, amount, hour, amount_mean, amount_std
        )
    else:
        composite = simple
    labels, actions = zip(*(categorize_risk(s) for s in simple))
    return pd.DataFrame(
        {
            "fraud_probability": np.asarray(fraud_proba, dtype=float),
            "risk_score": simple,
            "composite_risk_score": composite,
            "risk_band": list(labels),
            "recommended_action": list(actions),
        }
    )
