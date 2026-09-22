import numpy as np

from fraud_detection.evaluate import metrics_from_scores, optimize_threshold, pr_auc_score


def test_perfect_scores():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.99, 0.98])
    m = metrics_from_scores(y, p, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert pr_auc_score(y, p) == 1.0


def test_all_negative_predictions_have_zero_recall():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.1, 0.1])
    m = metrics_from_scores(y, p, threshold=0.5)
    assert m["recall"] == 0.0
    assert m["tp"] == 0


def test_threshold_maximises_f1():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.20, 0.40, 0.55, 0.80, 0.95])
    info = optimize_threshold(y, p)
    assert 0.0 < info["threshold"] <= 0.80
    assert info["f1"] > 0.5
