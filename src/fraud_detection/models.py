"""Train baseline + boosting models with class-imbalance handling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraud_detection.config import RANDOM_STATE


def scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return n_neg / n_pos


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 0.10,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample fraud on the *training* split only.

    sampling_strategy=0.10 → minority ends up at 10% of the majority (not a full 1:1
    duplicate of hundreds of thousands of synthetic rows).
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    current_ratio = n_pos / max(n_neg, 1)
    if n_pos < 6 or current_ratio >= sampling_strategy:
        print("STEP 3: SMOTE skipped (too few fraud rows, or already above target ratio)")
        return X, y
    k = max(1, min(5, n_pos - 1))
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name="Class")
    print("STEP 3: SMOTE (training set only)")
    print(f"  Before : legit={(y == 0).sum():,}  fraud={(y == 1).sum():,}")
    print(f"  After  : legit={(y_res == 0).sum():,}  fraud={(y_res == 1).sum():,}")
    return X_res, y_res


@dataclass
class TrainedModel:
    name: str
    estimator: object
    used_smote: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def predict_proba(self, X) -> np.ndarray:
        proba = self.estimator.predict_proba(X)
        return np.asarray(proba)[:, 1]


class SoftVoteEnsemble:
    """Average fraud probabilities from several fitted estimators."""

    def __init__(self, members: dict[str, object]) -> None:
        self.members = members

    def predict_proba(self, X) -> np.ndarray:
        stacked = np.column_stack(
            [np.asarray(m.predict_proba(X))[:, 1] for m in self.members.values()]
        )
        avg = stacked.mean(axis=1)
        return np.column_stack([1.0 - avg, avg])


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    print("MODEL: Logistic Regression (class_weight='balanced' + SMOTE train set)")
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, fast: bool = False) -> RandomForestClassifier:
    print("MODEL: Random Forest (class_weight='balanced')")
    model = RandomForestClassifier(
        n_estimators=80 if fast else 200,
        max_depth=12 if fast else 16,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val, y_val, fast: bool = False) -> XGBClassifier:
    print("MODEL: XGBoost (scale_pos_weight, eval_metric=aucpr)")
    spw = scale_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=120 if fast else 500,
        max_depth=4 if fast else 5,
        learning_rate=0.08 if fast else 0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_lambda=1.5,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=20 if fast else 40,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best = getattr(model, "best_iteration", None)
    print(f"  scale_pos_weight={spw:.1f}  best_iteration={best}")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, fast: bool = False) -> LGBMClassifier:
    print("MODEL: LightGBM (scale_pos_weight, eval_metric=average_precision)")
    spw = scale_pos_weight(y_train)
    model = LGBMClassifier(
        n_estimators=120 if fast else 500,
        learning_rate=0.08 if fast else 0.05,
        num_leaves=24 if fast else 31,
        max_depth=5 if fast else 6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        scale_pos_weight=spw,
        objective="binary",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            early_stopping(20 if fast else 40, verbose=False),
            log_evaluation(0),
        ],
    )
    best = getattr(model, "best_iteration_", None)
    print(f"  scale_pos_weight={spw:.1f}  best_iteration={best}")
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train_smote: pd.DataFrame,
    y_train_smote: pd.Series,
    fast: bool = False,
) -> dict[str, TrainedModel]:
    print("=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)

    lr = train_logistic_regression(X_train_smote, y_train_smote)
    rf = train_random_forest(X_train, y_train, fast=fast)
    xgb = train_xgboost(X_train, y_train, X_val, y_val, fast=fast)
    lgbm = train_lightgbm(X_train, y_train, X_val, y_val, fast=fast)

    ensemble = SoftVoteEnsemble({"xgboost": xgb, "lightgbm": lgbm, "random_forest": rf})

    return {
        "Logistic Regression": TrainedModel(
            "Logistic Regression", lr, used_smote=True, notes="balanced weights + SMOTE"
        ),
        "Random Forest": TrainedModel(
            "Random Forest", rf, notes="class_weight=balanced_subsample"
        ),
        "XGBoost": TrainedModel(
            "XGBoost", xgb, notes="scale_pos_weight + early stopping on PR-AUC"
        ),
        "LightGBM": TrainedModel(
            "LightGBM", lgbm, notes="scale_pos_weight + early stopping on AP"
        ),
        "Ensemble (soft vote)": TrainedModel(
            "Ensemble (soft vote)",
            ensemble,
            notes="mean(P) of XGBoost + LightGBM + Random Forest",
        ),
    }
