import numpy as np

from fraud_detection.risk import (
    categorize_risk,
    composite_risk_score,
    simple_risk_score,
)


def test_simple_risk_scales_probability():
    scores = simple_risk_score(np.array([0.0, 0.31, 0.79, 1.0]))
    assert scores.tolist() == [0.0, 31.0, 79.0, 100.0]
    assert categorize_risk(10)[0] == "Low"
    assert categorize_risk(45)[0] == "Medium"
    assert categorize_risk(70)[0] == "High"
    assert categorize_risk(90)[0] == "Critical"


def test_composite_raises_night_and_outlier_amount():
    proba = np.array([0.10, 0.10])
    amount = np.array([88.0, 5000.0])
    hour = np.array([14.0, 2.0])  # afternoon vs night
    scores = composite_risk_score(proba, amount, hour, amount_mean=88.0, amount_std=50.0)
    assert scores[1] > scores[0]
