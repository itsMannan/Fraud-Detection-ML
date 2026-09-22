# Fraud detection — training report

## Dataset

| | |
|---|---|
| Transactions | 284,807 |
| Legitimate | 284,315 |
| Fraud | 492 |
| Fraud rate | 0.173% |
| Imbalance | 1:578 |
| Missing values | 0 |

Accuracy is **not** used. A dummy model that always predicts legitimate would
score 99.827% accuracy and catch **zero** fraud.

## Champion model

**XGBoost**  ·  decision threshold **0.8957**
(F1-optimal on the validation split, frozen before the hold-out test set).

| Metric | Test score | Target | Status |
|---|---|---|---|
| precision | 0.9286 | 0.75–0.85 | MET |
| recall | 0.7959 | 0.80–0.90 | NEAR |
| f1 | 0.8571 | 0.75–0.85 | MET |
| pr_auc | 0.8794 | 0.80+ | MET |
| roc_auc | 0.9805 | secondary | — |
| specificity | 0.9999 | — | — |
| balanced_accuracy | 0.8979 | — | — |
| MCC | 0.8595 | — | — |

### Confusion matrix (test)

|  | Pred legitimate | Pred fraud |
|---|---|---|
| Actual legitimate | TN 56,858 | FP 6 |
| Actual fraud | FN 20 | TP 78 |

Recall 79.59% of fraud was caught.
Precision 92.86% of fraud flags were correct.

Validation F1-optimal threshold search: precision=0.9574,
recall=0.7627, F1=0.8491.

## All models (same hold-out test set)

| model | threshold | precision | recall | f1 | pr_auc | roc_auc | specificity | balanced_accuracy | mcc | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.8957 | 0.9286 | 0.7959 | 0.8571 | 0.8794 | 0.9805 | 0.9999 | 0.8979 | 0.8595 | 78 | 6 | 20 | 56858 |
| Ensemble (soft vote) | 0.7538 | 0.9615 | 0.7653 | 0.8523 | 0.8724 | 0.9791 | 0.9999 | 0.8826 | 0.8576 | 75 | 3 | 23 | 56861 |
| Random Forest | 0.3880 | 0.8804 | 0.8265 | 0.8526 | 0.8638 | 0.9654 | 0.9998 | 0.9132 | 0.8528 | 81 | 11 | 17 | 56853 |
| Logistic Regression | 0.9650 | 0.7232 | 0.8265 | 0.7714 | 0.7454 | 0.9734 | 0.9995 | 0.9130 | 0.7727 | 81 | 31 | 17 | 56833 |

## Stratified cross-validation (training split)

| model | precision | precision_std | recall | recall_std | f1 | f1_std | roc_auc | roc_auc_std | pr_auc | pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.0572 | 0.0049 | 0.8627 | 0.0697 | 0.1071 | 0.0084 | 0.9562 | 0.0343 | 0.7494 | 0.0618 |
| Random Forest | 0.8883 | 0.0173 | 0.7460 | 0.0612 | 0.8097 | 0.0369 | 0.9468 | 0.0375 | 0.8206 | 0.0544 |
| XGBoost | 0.8401 | 0.0244 | 0.7971 | 0.0485 | 0.8173 | 0.0308 | 0.9853 | 0.0154 | 0.8483 | 0.0672 |

## Risk bands

| Score | Band | Action |
|---|---|---|
| 0–30 | Low | Approve immediately |
| 30–60 | Medium | Manual review recommended |
| 60–80 | High | Block and notify customer |
| 80–100 | Critical | Block and escalate to fraud team |

Risk score = P(fraud) × 100. A composite score also blends amount deviation and night-time flags.
