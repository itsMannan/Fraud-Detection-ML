#!/usr/bin/env python3
"""Score raw credit-card transactions with the saved champion model.

Examples
--------
python predict.py --csv examples/sample_transactions.csv
python predict.py --json '{"Time": 0, "Amount": 149.62, "V1": -1.36, ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.config import MODELS  # noqa: E402
from fraud_detection.persist import load_bundle  # noqa: E402
from fraud_detection.risk import score_frame  # noqa: E402


def _proba(model, X):
    raw = model.predict_proba(X)
    return raw[:, 1]


def score_frame_with_bundle(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    raw = df.copy()
    raw.columns = [str(c).strip().strip('"') for c in raw.columns]
    if "Class" in raw.columns:
        raw = raw.drop(columns=["Class"])

    fe = bundle["engineer"].transform(raw)
    missing = [c for c in bundle["feature_names"] if c not in fe.columns]
    if missing:
        raise ValueError(
            "Input is missing engineered/raw features required by the model: "
            + ", ".join(missing)
        )
    X = fe[bundle["feature_names"]]
    Xs = bundle["scaler"].transform(X)
    proba = _proba(bundle["model"], Xs)

    amount = fe["Amount"].to_numpy() if "Amount" in fe.columns else None
    hour = fe["Hour"].to_numpy() if "Hour" in fe.columns else None
    scored = score_frame(
        proba,
        amount=amount,
        hour=hour,
        amount_mean=bundle["engineer"].amount_mean_,
        amount_std=bundle["engineer"].amount_std_,
    )
    scored.insert(0, "flagged_as_fraud", (proba >= bundle["threshold"]).astype(int))
    scored.insert(1, "decision_threshold", bundle["threshold"])
    return scored


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score transactions for fraud risk.")
    p.add_argument("--csv", type=str, help="CSV of raw transactions (ULB columns).")
    p.add_argument("--json", type=str, help="JSON object or list of objects.")
    p.add_argument(
        "--model",
        type=str,
        default=str(MODELS / "champion_bundle.joblib"),
        help="Path to champion_bundle.joblib",
    )
    p.add_argument("--out", type=str, default=None, help="Optional CSV to write.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    bundle = load_bundle(Path(args.model))
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.json:
        payload = json.loads(args.json)
        df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    else:
        raise SystemExit("Provide --csv or --json")

    scored = score_frame_with_bundle(df, bundle)
    print(f"Model: {bundle['model_name']}  threshold={bundle['threshold']:.4f}")
    with pd.option_context("display.max_columns", 20, "display.width", 140):
        print(scored.to_string(index=False))
    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
