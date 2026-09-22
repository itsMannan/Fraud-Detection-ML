"""Imbalanced-classification metrics, threshold search, and CV."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from fraud_detection.config import N_CV_FOLDS, RANDOM_STATE


def pr_auc_score(y_true, y_score) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, y_score))


def metrics_from_scores(
    y_true,
    y_score,
    threshold: float = 0.5,
    target_names: tuple[str, str] = ("Legitimate", "Fraud"),
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = pr_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if (tp + fp + fn + tn) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_pred_positive": int(y_pred.sum()),
        "n_true_positive": int(y_true.sum()),
        "classification_report": classification_report(
            y_true, y_pred, target_names=list(target_names), zero_division=0
        ),
    }


def optimize_threshold(
    y_true,
    y_score,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> dict[str, float]:
    """Pick the probability cutoff that maximises F1 on a validation fold.

    Optionally constrain the search to a minimum precision or recall.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one extra element vs thresholds
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)

    mask = np.ones_like(f1, dtype=bool)
    if min_precision is not None:
        mask &= precision[:-1] >= min_precision
    if min_recall is not None:
        mask &= recall[:-1] >= min_recall
    if not mask.any():
        mask = np.ones_like(f1, dtype=bool)

    idx = int(np.nanargmax(np.where(mask, f1, -np.inf)))
    best_t = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return {
        "threshold": best_t,
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
    }


def print_metrics(name: str, metrics: dict, header: str | None = None) -> None:
    title = header or f"EVALUATION: {name}"
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"  Threshold     {metrics['threshold']:.4f}")
    print(f"  Precision     {metrics['precision']:.4f}   (target 0.75–0.85)")
    print(f"  Recall        {metrics['recall']:.4f}   (target 0.80–0.90)")
    print(f"  F1            {metrics['f1']:.4f}   (target 0.75–0.85)")
    print(f"  PR-AUC        {metrics['pr_auc']:.4f}   (target 0.80+)  ← primary")
    print(f"  ROC-AUC       {metrics['roc_auc']:.4f}")
    print(f"  Specificity   {metrics['specificity']:.4f}")
    print(f"  Balanced acc  {metrics['balanced_accuracy']:.4f}")
    print(f"  MCC           {metrics['mcc']:.4f}")
    print(
        "  Confusion     "
        f"TN={metrics['tn']:,}  FP={metrics['fp']:,}  "
        f"FN={metrics['fn']:,}  TP={metrics['tp']:,}"
    )
    print(metrics["classification_report"])


def stratified_cv_table(
    estimators: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_CV_FOLDS,
) -> pd.DataFrame:
    """Stratified k-fold scores. PR-AUC is `average_precision` in sklearn."""
    print("=" * 70)
    print(f"STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    rows = []
    for name, est in estimators.items():
        print(f"  CV {name}...")
        cv = cross_validate(
            est,
            X,
            y,
            cv=skf,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        row = {"model": name}
        for key in scoring:
            row[key] = float(cv[f"test_{key}"].mean())
            row[f"{key}_std"] = float(cv[f"test_{key}"].std())
            print(
                f"    {key:10s} {row[key]:.4f} ± {row[f'{key}_std']:.4f}"
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
