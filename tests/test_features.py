import numpy as np
import pandas as pd

from fraud_detection.features import FeatureEngineer, split_xy


def test_engineer_adds_temporal_and_amount_features():
    df = pd.DataFrame(
        {
            "Time": [0, 3600, 50_000],
            "V1": [0.1, -0.2, 0.3],
            "Amount": [0.0, 10.0, 250.0],
            "Class": [0, 0, 1],
        }
    )
    fe = FeatureEngineer().fit(df)
    out = fe.transform(df)
    for col in ("Amount_log", "Amount_zscore", "Hour", "Hour_sin", "Hour_cos", "Is_night"):
        assert col in out.columns
    assert np.isclose(out["Amount_log"].iloc[0], 0.0)
    assert out["Is_night"].iloc[0] == 1  # midnight
    X, y = split_xy(out)
    assert "Class" not in X.columns
    assert y.tolist() == [0, 0, 1]


def test_amount_stats_come_from_train_only():
    train = pd.DataFrame({"Amount": [10.0, 10.0, 10.0], "V1": [0, 1, 2], "Class": [0, 0, 1]})
    test = pd.DataFrame({"Amount": [1000.0], "V1": [3], "Class": [0]})
    fe = FeatureEngineer().fit(train)
    assert np.isclose(fe.amount_mean_, 10.0)
    transformed = fe.transform(test)
    # z-score uses train mean/std, not the test amount itself
    assert transformed["Amount_zscore"].iloc[0] > 0
