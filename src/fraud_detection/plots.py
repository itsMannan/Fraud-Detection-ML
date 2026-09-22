"""Publication-quality figures for the README and the training report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from fraud_detection.config import FIGURES, TARGETS

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PALETTE = {
    "legit": "#2b6cb0",
    "fraud": "#c53030",
    "accent": "#2f855a",
    "warn": "#c05621",
    "header": "#1a365d",
    "muted": "#4a5568",
}
MODEL_COLORS = [
    "#2b6cb0",
    "#2f855a",
    "#c05621",
    "#6b46c1",
    "#c53030",
    "#2c7a7b",
]


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    counts = df["Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        ["Legitimate (0)", "Fraud (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=[PALETTE["legit"], PALETTE["fraud"]],
        width=0.55,
    )
    for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}\n({val / len(df) * 100:.3f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("Transactions")
    ax.set_title("Class imbalance — accuracy is the wrong metric")
    ax.set_ylim(0, counts.get(0, 0) * 1.12)
    fig.text(
        0.5,
        -0.02,
        "A dummy “always legitimate” model would score 99.83% accuracy and catch 0 fraud.",
        ha="center",
        color=PALETTE["muted"],
        fontsize=9,
    )
    return _save(fig, "class_balance", out_dir)


def plot_amount_by_class(df: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    tmp = df[["Amount", "Class"]].copy()
    tmp["log_amount"] = np.log1p(tmp["Amount"].clip(lower=0))
    sns.kdeplot(
        data=tmp[tmp["Class"] == 0],
        x="log_amount",
        ax=ax,
        fill=True,
        color=PALETTE["legit"],
        label="Legitimate",
        alpha=0.45,
    )
    sns.kdeplot(
        data=tmp[tmp["Class"] == 1],
        x="log_amount",
        ax=ax,
        fill=True,
        color=PALETTE["fraud"],
        label="Fraud",
        alpha=0.45,
    )
    ax.set_xlabel("log1p(Amount)")
    ax.set_title("Transaction amount distribution by class")
    ax.legend()
    return _save(fig, "amount_by_class", out_dir)


def plot_feature_correlations(
    df: pd.DataFrame, out_dir: Path = FIGURES, top_n: int = 16
) -> Path | None:
    numeric = df.select_dtypes(include=[np.number])
    if "Class" not in numeric.columns:
        return None
    corr = numeric.corr()["Class"].drop(labels=["Class"]).abs().sort_values(ascending=False)
    top = corr.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = [PALETTE["fraud"] if v > 0.1 else PALETTE["legit"] for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("|Pearson correlation| with Class")
    ax.set_title("Features most associated with fraud (train split)")
    return _save(fig, "feature_correlations", out_dir)


def plot_pr_curves(results: dict, y_true, out_dir: Path = FIGURES) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, color="#a0aec0", ls="--", lw=1, label=f"Baseline (prevalence={prevalence:.4f})")
    for (name, payload), color in zip(results.items(), MODEL_COLORS):
        p, r, _ = precision_recall_curve(y_true, payload["proba"])
        ax.plot(r, p, color=color, lw=2, label=f"{name}  (AP={payload['metrics']['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–Recall curves (hold-out test set)")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, "pr_curves", out_dir)


def plot_roc_curves(results: dict, y_true, out_dir: Path = FIGURES) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.plot([0, 1], [0, 1], color="#a0aec0", ls="--", lw=1, label="Chance")
    for (name, payload), color in zip(results.items(), MODEL_COLORS):
        fpr, tpr, _ = roc_curve(y_true, payload["proba"])
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{name}  (AUC={payload['metrics']['roc_auc']:.3f})",
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (secondary metric — less informative under imbalance)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, "roc_curves", out_dir)


def plot_confusion_matrices(results: dict, y_true, out_dir: Path = FIGURES) -> Path:
    n = len(results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    y_true = np.asarray(y_true)
    for ax, (name, payload) in zip(axes, results.items()):
        m = payload["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred legit", "Pred fraud"],
            yticklabels=["True legit", "True fraud"],
        )
        ax.set_title(f"{name}\nF1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices @ F1-optimal threshold (test set)", y=1.02)
    fig.tight_layout()
    return _save(fig, "confusion_matrices", out_dir)


def plot_metrics_comparison(results: dict, out_dir: Path = FIGURES) -> Path:
    labels = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    pretty = ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(labels))
    width = 0.14
    names = list(results)
    for i, name in enumerate(names):
        vals = [results[name]["metrics"][k] for k in labels]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=name, color=MODEL_COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(pretty)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Hold-out test metrics (do not use accuracy)")
    ax.axhline(0.80, color=PALETTE["accent"], ls=":", lw=1, label="0.80 target")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    return _save(fig, "metrics_comparison", out_dir)


def plot_threshold_curve(y_true, y_score, chosen: float, out_dir: Path = FIGURES) -> Path:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(thresholds, precision[:-1], label="Precision", color=PALETTE["legit"])
    ax.plot(thresholds, recall[:-1], label="Recall", color=PALETTE["fraud"])
    ax.plot(thresholds, f1, label="F1", color=PALETTE["accent"])
    ax.axvline(chosen, color=PALETTE["warn"], ls="--", label=f"Chosen t={chosen:.3f}")
    ax.set_xlabel("Fraud-probability threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep on the validation split (F1-optimal cutoff locked before test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, "threshold_analysis", out_dir)


def plot_feature_importance(model, feature_names, out_dir: Path = FIGURES, top_n: int = 18) -> Path | None:
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_))
    if importance is None or len(importance) != len(feature_names):
        return None
    order = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(np.array(feature_names)[order], importance[order], color=PALETTE["header"])
    ax.set_xlabel("Importance")
    ax.set_title("Best-model feature importance (risk-score drivers)")
    return _save(fig, "feature_importance", out_dir)


def plot_risk_distribution(risk_df: pd.DataFrame, y_true, out_dir: Path = FIGURES) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    tmp = risk_df.copy()
    tmp["Class"] = np.asarray(y_true)
    sns.histplot(
        data=tmp,
        x="risk_score",
        hue="Class",
        bins=40,
        palette={0: PALETTE["legit"], 1: PALETTE["fraud"]},
        ax=axes[0],
        stat="density",
        common_norm=False,
        alpha=0.6,
    )
    axes[0].set_title("Risk score by true class")
    axes[0].set_xlabel("Risk score (0–100)")

    order = ["Low", "Medium", "High", "Critical"]
    counts = (
        tmp.groupby(["risk_band", "Class"])
        .size()
        .unstack(fill_value=0)
        .reindex(order)
        .fillna(0)
    )
    counts.plot(
        kind="bar",
        ax=axes[1],
        color=[PALETTE["legit"], PALETTE["fraud"]],
        rot=0,
    )
    axes[1].set_title("Risk band vs. true class (test set)")
    axes[1].set_xlabel("Risk band")
    axes[1].set_ylabel("Transactions")
    axes[1].legend(["Legitimate", "Fraud"], title=None)
    fig.tight_layout()
    return _save(fig, "risk_score_distribution", out_dir)


def plot_scorecard(
    best_name: str,
    metrics: dict,
    comparison: pd.DataFrame,
    out_dir: Path = FIGURES,
) -> Path:
    """Dashboard-style screenshot of hold-out scores for the README."""
    fig = plt.figure(figsize=(14.5, 8.6))
    fig.patch.set_facecolor("#f7fafc")
    gs = fig.add_gridspec(3, 4, height_ratios=[0.55, 1.15, 1.35], hspace=0.42, wspace=0.28)

    header = fig.add_subplot(gs[0, :])
    header.set_xlim(0, 1)
    header.set_ylim(0, 1)
    header.axis("off")
    header.add_patch(plt.Rectangle((0, 0), 1, 1, color=PALETTE["header"], transform=header.transAxes))
    header.text(
        0.02,
        0.62,
        "Intelligent Credit Card Fraud Detection  ·  Hold-out test scores",
        color="white",
        fontsize=16,
        fontweight="bold",
        va="center",
    )
    header.text(
        0.02,
        0.22,
        f"Champion model: {best_name}    ·    Accuracy is not reported (0.17% fraud rate)",
        color="#bee3f8",
        fontsize=11,
        va="center",
    )

    tiles = [
        ("Precision", metrics["precision"], TARGETS["precision"]),
        ("Recall", metrics["recall"], TARGETS["recall"]),
        ("F1 score", metrics["f1"], TARGETS["f1"]),
        ("PR-AUC", metrics["pr_auc"], TARGETS["pr_auc"]),
    ]
    for i, (label, value, (lo, hi)) in enumerate(tiles):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        hit = value >= lo
        edge = PALETTE["accent"] if hit else PALETTE["warn"]
        ax.add_patch(
            plt.Rectangle(
                (0.04, 0.08),
                0.92,
                0.84,
                fill=True,
                facecolor="white",
                edgecolor=edge,
                lw=2.5,
                transform=ax.transAxes,
            )
        )
        ax.text(0.5, 0.78, label, ha="center", va="center", fontsize=12, color=PALETTE["muted"])
        ax.text(
            0.5,
            0.46,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color=PALETTE["header"],
        )
        status = f"Target {lo:.2f}–{hi:.2f}  ·  {'MET' if hit else 'NEAR'}"
        if label == "PR-AUC":
            status = f"Target {lo:.2f}+  ·  {'MET' if hit else 'NEAR'}"
        ax.text(
            0.5,
            0.18,
            status,
            ha="center",
            va="center",
            fontsize=9,
            color=edge,
            fontweight="bold",
        )

    ax_cm = fig.add_subplot(gs[2, 0:2])
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        ax=ax_cm,
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Predicted legitimate", "Predicted fraud"],
        yticklabels=["Actual legitimate", "Actual fraud"],
    )
    ax_cm.set_title(
        f"Confusion matrix  ·  threshold={metrics['threshold']:.3f}  ·  "
        f"MCC={metrics['mcc']:.3f}"
    )

    ax_tbl = fig.add_subplot(gs[2, 2:4])
    ax_tbl.axis("off")
    cols = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    cell = comparison[cols].copy()
    cell = cell.round(3)
    table = ax_tbl.table(
        cellText=cell.values,
        rowLabels=list(cell.index),
        colLabels=["Prec", "Rec", "F1", "PR-AUC", "ROC-AUC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.55)
    for (r, c), cell_obj in table.get_celld().items():
        if r == 0:
            cell_obj.set_facecolor(PALETTE["header"])
            cell_obj.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell_obj.set_facecolor("#edf2f7")
    ax_tbl.set_title("All models on the same hold-out test set", pad=12)

    return _save(fig, "scorecard", out_dir)


def plot_calibration(y_true, y_score, out_dir: Path = FIGURES) -> Path:
    from sklearn.calibration import calibration_curve

    frac, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot([0, 1], [0, 1], "--", color="#a0aec0", label="Perfect calibration")
    ax.plot(mean_pred, frac, "o-", color=PALETTE["header"], label="Champion model")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Reliability diagram (test set)")
    ax.legend()
    return _save(fig, "calibration", out_dir)
