# src/ml/data/ — Data collection and preparation
#
# psd2_collector.py    → fetches real transactions from GoCardless (PSD2 Open Banking)
# label_transactions.py → rule-based auto-labeling (REMA 1000 → "groceries")
# synthetic_generator.py → generates realistic Norwegian transactions for local testing
# prepare_dataset.py   → combines sources, labels, splits 70/15/15, saves Parquet
