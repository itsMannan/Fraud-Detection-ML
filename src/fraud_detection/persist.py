"""Serialize the champion bundle and write JSON/Markdown reports."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.config import MODELS, REPORTS, TARGETS


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as GitHub-flavored markdown without the tabulate extra."""
    table = df.copy()
    if table.index.name or not all(i == n for n, i in enumerate(table.index)):
        table = table.reset_index()
    cols = [str(c) for c in table.columns]

    def fmt(val) -> str:
        if isinstance(val, (float, np.floating)):
            return f"{float(val):.4f}"
        return str(val)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body])


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def save_bundle(bundle: dict, path: Path | None = None) -> Path:
    path = path or (MODELS / "champion_bundle.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"Saved model bundle → {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def load_bundle(path: Path | None = None) -> dict:
    path = path or (MODELS / "champion_bundle.joblib")
    return joblib.load(path)


def target_status(metric: str, value: float) -> str:
    lo, hi = TARGETS[metric]
    if metric == "pr_auc":
        return "MET" if value >= lo else "BELOW"
    if lo <= value:
        return "MET" if value <= hi + 0.05 else "ABOVE RANGE"
    return "BELOW"


def write_summary_markdown(
    dataset_stats: dict,
    best_name: str,
    test_metrics: dict,
    comparison: pd.DataFrame,
    threshold_info: dict,
    cv_table: pd.DataFrame | None,
    path: Path | None = None,
) -> Path:
    path = path or (REPORTS / "summary.md")
    path.parent.mkdir(parents=True, exist_ok=True)

    def pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    rows = []
    for metric in ("precision", "recall", "f1", "pr_auc"):
        val = test_metrics[metric]
        lo, hi = TARGETS[metric]
        target = f"{lo:.2f}+" if metric == "pr_auc" else f"{lo:.2f}–{hi:.2f}"
        rows.append(
            f"| {metric} | {val:.4f} | {target} | {target_status(metric, val)} |"
        )

    cv_block = ""
    if cv_table is not None and len(cv_table):
        cv_block = "\n## Stratified cross-validation (training split)\n\n"
        cv_block += dataframe_to_markdown(cv_table.round(4)) + "\n"

    comparison_md = dataframe_to_markdown(comparison.round(4))

    text = f"""# Fraud detection — training report

## Dataset

| | |
|---|---|
| Transactions | {dataset_stats['n_rows']:,} |
| Legitimate | {dataset_stats['n_legitimate']:,} |
| Fraud | {dataset_stats['n_fraud']:,} |
| Fraud rate | {dataset_stats['fraud_rate'] * 100:.3f}% |
| Imbalance | 1:{dataset_stats['imbalance_ratio']:.0f} |
| Missing values | {dataset_stats['missing_values']} |

Accuracy is **not** used. A dummy model that always predicts legitimate would
score {100 - dataset_stats['fraud_rate'] * 100:.3f}% accuracy and catch **zero** fraud.

## Champion model

**{best_name}**  ·  decision threshold **{test_metrics['threshold']:.4f}**
(F1-optimal on the validation split, frozen before the hold-out test set).

| Metric | Test score | Target | Status |
|---|---|---|---|
{chr(10).join(rows)}
| roc_auc | {test_metrics['roc_auc']:.4f} | secondary | — |
| specificity | {test_metrics['specificity']:.4f} | — | — |
| balanced_accuracy | {test_metrics['balanced_accuracy']:.4f} | — | — |
| MCC | {test_metrics['mcc']:.4f} | — | — |

### Confusion matrix (test)

|  | Pred legitimate | Pred fraud |
|---|---|---|
| Actual legitimate | TN {test_metrics['tn']:,} | FP {test_metrics['fp']:,} |
| Actual fraud | FN {test_metrics['fn']:,} | TP {test_metrics['tp']:,} |

Recall {pct(test_metrics['recall'])} of fraud was caught.
Precision {pct(test_metrics['precision'])} of fraud flags were correct.

Validation F1-optimal threshold search: precision={threshold_info.get('precision', float('nan')):.4f},
recall={threshold_info.get('recall', float('nan')):.4f}, F1={threshold_info.get('f1', float('nan')):.4f}.

## All models (same hold-out test set)

{comparison_md}
{cv_block}
## Risk bands

| Score | Band | Action |
|---|---|---|
| 0–30 | Low | Approve immediately |
| 30–60 | Medium | Manual review recommended |
| 60–80 | High | Block and notify customer |
| 80–100 | Critical | Block and escalate to fraud team |

Risk score = P(fraud) × 100. A composite score also blends amount deviation and night-time flags.
"""
    path.write_text(text, encoding="utf-8")
    return path
