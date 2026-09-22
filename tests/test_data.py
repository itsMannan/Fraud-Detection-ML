from fraud_detection.data import make_synthetic_dataset


def test_synthetic_dataset_is_imbalanced_and_labelled():
    df = make_synthetic_dataset(n_samples=1000, n_features=8, fraud_rate=0.03)
    assert "Class" in df.columns
    assert "Amount" in df.columns
    assert "Time" in df.columns
    assert set(df["Class"].unique()) <= {0, 1}
    fraud_rate = df["Class"].mean()
    assert 0.005 < fraud_rate < 0.15
    assert len(df) == 1000
