"""End-to-end training pipeline: split → engineer → train → evaluate → persist."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection.config import (
    ARTIFACTS,
    CV_SUBSAMPLE,
    EXAMPLES,
    FIGURES,
    MODELS,
    N_CV_FOLDS,
    RANDOM_STATE,
    REPORTS,
    TEST_SIZE,
    VAL_SIZE,
)
from fraud_detection.data import describe_dataset, load_transactions, make_synthetic_dataset
from fraud_detection.evaluate import (
    metrics_from_scores,
    optimize_threshold,
    print_metrics,
    stratified_cv_table,
)
from fraud_detection.features import FeatureEngineer, split_xy
from fraud_detection.models import apply_smote, scale_pos_weight, train_all_models
from fraud_detection.persist import save_bundle, save_json, write_summary_markdown
from fraud_detection.plots import (
    plot_amount_by_class,
    plot_calibration,
    plot_class_balance,
    plot_confusion_matrices,
    plot_feature_correlations,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_pr_curves,
    plot_risk_distribution,
    plot_roc_curves,
    plot_scorecard,
    plot_threshold_curve,
)
from fraud_detection.risk import score_frame


def _ensure_dirs() -> None:
    for path in (ARTIFACTS, FIGURES, MODELS, REPORTS, EXAMPLES):
        path.mkdir(parents=True, exist_ok=True)


def _comparison_frame(results: dict) -> pd.DataFrame:
    rows = []
    for name, payload in results.items():
        m = payload["metrics"]
        rows.append(
            {
                "model": name,
                "threshold": m["threshold"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "specificity": m["specificity"],
                "balanced_accuracy": m["balanced_accuracy"],
                "mcc": m["mcc"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False)


def _subsample_stratified(X: pd.DataFrame, y: pd.Series, n: int, random_state: int):
    if len(X) <= n:
        return X, y
    frac = n / len(X)
    splitter = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=random_state
    )
    return splitter[0], splitter[2]


def run_pipeline(
    data_path: str | Path | None = None,
    demo: bool = False,
    fast: bool = False,
    skip_cv: bool = False,
) -> dict:
    """Run the full fraud-detection pipeline and write artifacts."""
    t0 = time.time()
    _ensure_dirs()

    print("\n" + "=" * 70)
    print("INTELLIGENT CREDIT CARD FRAUD DETECTION & RISK SCORING")
    print("=" * 70)

    if demo:
        df = make_synthetic_dataset(n_samples=6_000 if fast else 12_000)
        print("Using synthetic demo data (not the ULB dataset).")
    else:
        df = load_transactions(data_path)

    stats = describe_dataset(df)

    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING & STRATIFIED SPLITS")
    print("=" * 70)

    df_trainval, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["Class"], random_state=RANDOM_STATE
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=VAL_SIZE,
        stratify=df_trainval["Class"],
        random_state=RANDOM_STATE,
    )
    for split_name, split_df in (("train", df_train), ("val", df_val), ("test", df_test)):
        rate = split_df["Class"].mean() * 100
        print(
            f"  {split_name:5s}: {len(split_df):,} rows  "
            f"fraud={int(split_df['Class'].sum()):,} ({rate:.3f}%)"
        )

    engineer = FeatureEngineer().fit(df_train)
    train_fe = engineer.transform(df_train)
    val_fe = engineer.transform(df_val)
    test_fe = engineer.transform(df_test)

    X_train, y_train = split_xy(train_fe)
    X_val, y_val = split_xy(val_fe)
    X_test, y_test = split_xy(test_fe)
    feature_names = list(X_train.columns)

    # Keep raw amount/hour for composite risk scores (pre-scaling).
    test_amount = test_fe["Amount"].to_numpy() if "Amount" in test_fe.columns else None
    test_hour = test_fe["Hour"].to_numpy() if "Hour" in test_fe.columns else None

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=feature_names, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    print(f"  Engineered features ({len(feature_names)}): {feature_names}")
    print("  StandardScaler fitted on the training split only.")

    plot_class_balance(df)
    if "Amount" in df.columns:
        plot_amount_by_class(df)
    plot_feature_correlations(train_fe)

    X_smote, y_smote = apply_smote(X_train_s, y_train)

    models = train_all_models(
        X_train_s,
        y_train,
        X_val_s,
        y_val,
        X_smote,
        y_smote,
        fast=fast,
    )

    print("\n" + "=" * 70)
    print("STEP 5–6: VALIDATION THRESHOLDS + HOLD-OUT TEST EVALUATION")
    print("=" * 70)

    results: dict = {}
    for name, trained in models.items():
        val_proba = trained.predict_proba(X_val_s)
        thresh_info = optimize_threshold(y_val, val_proba)
        test_proba = trained.predict_proba(X_test_s)
        test_metrics = metrics_from_scores(y_test, test_proba, threshold=thresh_info["threshold"])
        print_metrics(name, test_metrics, header=f"TEST: {name}")
        results[name] = {
            "trained": trained,
            "val_threshold": thresh_info,
            "proba": test_proba,
            "metrics": test_metrics,
            "val_pr_auc": float(average_precision_score(y_val, val_proba)),
        }

    champion_name = max(results, key=lambda n: results[n]["val_pr_auc"])
    champion = results[champion_name]
    print(f"\nChampion by validation PR-AUC: {champion_name} "
          f"({champion['val_pr_auc']:.4f})")

    comparison = _comparison_frame(results)
    comparison.to_csv(REPORTS / "model_comparison.csv")

    plot_pr_curves(results, y_test)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_metrics_comparison(results)
    plot_threshold_curve(y_val, champion["trained"].predict_proba(X_val_s), champion["metrics"]["threshold"])
    plot_feature_importance(models["XGBoost"].estimator, feature_names)
    plot_calibration(y_test, champion["proba"])
    plot_scorecard(champion_name, champion["metrics"], comparison)

    print("\n" + "=" * 70)
    print("STEP 7: RISK SCORING")
    print("=" * 70)
    risk_df = score_frame(
        champion["proba"],
        amount=test_amount,
        hour=test_hour,
        amount_mean=engineer.amount_mean_,
        amount_std=engineer.amount_std_,
    )
    risk_df["y_true"] = y_test.to_numpy()
    risk_df.to_csv(REPORTS / "test_risk_scores.csv", index=False)
    print(risk_df["risk_band"].value_counts().to_string())
    print(f"  Mean risk score: {risk_df['risk_score'].mean():.2f}")
    plot_risk_distribution(risk_df, y_test)

    cv_table = None
    if not skip_cv:
        print("\n" + "=" * 70)
        print("STEP 8: STRATIFIED CROSS-VALIDATION (training split subsample)")
        print("=" * 70)
        spw = scale_pos_weight(y_train)
        n_est = 60 if fast else 100
        cv_estimators = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=n_est,
                max_depth=12,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=n_est,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                eval_metric="aucpr",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=n_est,
                learning_rate=0.1,
                num_leaves=31,
                scale_pos_weight=spw,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=-1,
            ),
        }
        X_cv, y_cv = _subsample_stratified(
            X_train_s, y_train, n=min(CV_SUBSAMPLE, 20_000 if fast else CV_SUBSAMPLE), random_state=RANDOM_STATE
        )
        folds = 3 if fast else N_CV_FOLDS
        cv_table = stratified_cv_table(cv_estimators, X_cv, y_cv, n_folds=folds)
        cv_table.to_csv(REPORTS / "cross_validation.csv")

    # Persist champion + preprocessing for predict.py / api.py
    bundle = {
        "model_name": champion_name,
        "model": champion["trained"].estimator,
        "engineer": engineer,
        "scaler": scaler,
        "feature_names": feature_names,
        "threshold": float(champion["metrics"]["threshold"]),
        "metrics": {k: v for k, v in champion["metrics"].items() if k != "classification_report"},
        "dataset_stats": stats,
    }
    save_bundle(bundle)

    metrics_payload = {
        "champion": champion_name,
        "dataset": stats,
        "test_metrics": bundle["metrics"],
        "validation_threshold_search": champion["val_threshold"],
        "all_models": {
            name: {
                "val_pr_auc": payload["val_pr_auc"],
                "val_threshold": payload["val_threshold"],
                "test": {k: v for k, v in payload["metrics"].items() if k != "classification_report"},
            }
            for name, payload in results.items()
        },
        "elapsed_seconds": time.time() - t0,
    }
    save_json(metrics_payload, REPORTS / "metrics.json")
    (REPORTS / "classification_report.txt").write_text(
        f"Champion: {champion_name}\n\n{champion['metrics']['classification_report']}",
        encoding="utf-8",
    )
    write_summary_markdown(
        stats,
        champion_name,
        champion["metrics"],
        comparison,
        champion["val_threshold"],
        cv_table,
    )

    # Example rows for the prediction CLI (mix of fraud + legit from the test split).
    labeled = df_test.copy()
    fraud_n = min(5, int((labeled["Class"] == 1).sum()))
    legit_n = min(5, int((labeled["Class"] == 0).sum()))
    sample = pd.concat(
        [
            labeled[labeled["Class"] == 1].head(fraud_n),
            labeled[labeled["Class"] == 0].head(legit_n),
        ]
    )
    sample.to_csv(EXAMPLES / "sample_transactions_labeled.csv", index=False)
    sample.drop(columns=["Class"]).to_csv(EXAMPLES / "sample_transactions.csv", index=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {elapsed / 60:.1f} min")
    print(f"  Champion : {champion_name}")
    print(f"  Precision: {champion['metrics']['precision']:.4f}")
    print(f"  Recall   : {champion['metrics']['recall']:.4f}")
    print(f"  F1       : {champion['metrics']['f1']:.4f}")
    print(f"  PR-AUC   : {champion['metrics']['pr_auc']:.4f}")
    print(f"  Figures  : {FIGURES}")
    print(f"  Reports  : {REPORTS}")
    print("=" * 70)

    return {
        "champion": champion_name,
        "metrics": champion["metrics"],
        "comparison": comparison,
        "results": results,
        "bundle": bundle,
        "stats": stats,
        "cv_table": cv_table,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the credit-card fraud detection and risk-scoring system."
    )
    parser.add_argument("--data", type=str, default=None, help="Path to creditcard.csv")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Train on a small synthetic table (no download).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fewer trees / fewer CV folds for a smoke run.",
    )
    parser.add_argument("--skip-cv", action="store_true", help="Skip stratified k-fold.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(data_path=args.data, demo=args.demo, fast=args.fast, skip_cv=args.skip_cv)


if __name__ == "__main__":
    main()
