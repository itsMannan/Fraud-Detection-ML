# Dataset

This project uses the public **ULB Machine Learning Group** credit-card fraud dataset
(284,807 transactions, 492 frauds, 0.172% positive class).

The copy committed as `creditcard-selected-columns.csv` at the repo root is an empty
placeholder (header only, no `Class` / `Amount` columns). Training therefore **auto-downloads**
the full public CSV into `data/creditcard.csv` (gitignored, ~144 MB).

## Automatic download

```bash
python train.py
```

`train.py` downloads from the TensorFlow public mirror if the file is missing:

https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

## Manual download

```bash
mkdir -p data
curl -L -o data/creditcard.csv \
  https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

Columns: `Time`, `V1`–`V28` (PCA-anonymized), `Amount`, `Class` (0 = legitimate, 1 = fraud).
