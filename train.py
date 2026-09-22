#!/usr/bin/env python3
"""Train the fraud-detection and risk-scoring system.

Examples
--------
python train.py
python train.py --data data/creditcard.csv
python train.py --fast --skip-cv
python train.py --demo --fast
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
