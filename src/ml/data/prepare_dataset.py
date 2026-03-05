"""
Dataset preparation pipeline for transaction categorization.

Combines data from multiple sources, labels it, and creates
train/validation/test splits ready for BERT fine-tuning.

Data sources (in order of quality):
  1. GoCardless PSD2 (real bank transactions) — highest quality
  2. Synthetic generator (realistic Norwegian transactions) — fills gaps
  3. [Future] Bronze layer ADLS (accumulated production events)

Output: Three Parquet files + fitted TF-IDF baseline model
  data/train.parquet      — 70% of labeled data → BERT learns from this
  data/val.parquet        — 15%                 → we watch this for overfitting
  data/test.parquet       — 15%                 → final honest accuracy report
  models/tfidf_vectorizer.joblib   — fitted TF-IDF (Layer 2 at inference time)
  models/tfidf_classifier.joblib   — fitted LogisticRegression (Layer 2)

Run this script from the project root:
  python -m src.ml.data.prepare_dataset --n-per-category 500

The Parquet files are then uploaded to Azure ML as a Data Asset,
so the training job (finetune_bert.py) can access them without local storage.

Key concepts in this file:
  - train/val/test split: why and how
  - class balance: why equal categories matter
  - TF-IDF: how it works, what "vectorizer" means
  - joblib: how we save Python objects to disk
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib                                    # Save/load Python objects to disk
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text → numbers
from sklearn.linear_model import LogisticRegression           # Simple ML classifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split          # Split into train/val/test
from sklearn.pipeline import Pipeline                         # Chain vectorizer + classifier

from src.ml.data.label_transactions import (
    CATEGORIES,
    CATEGORY_TO_ID,
    auto_label,
    get_coverage_stats,
)
from src.ml.data.synthetic_generator import generate_transactions, get_dataset_stats


# ─── Paths ────────────────────────────────────────────────────────────────────

# Project root — where requirements.txt and pyproject.toml live
PROJECT_ROOT = Path(__file__).parents[3]

# Where we save Parquet splits (uploaded to Azure ML as Data Asset)
DATA_DIR = PROJECT_ROOT / "data" / "ml"

# Where we save trained sklearn models (small enough for local storage)
MODELS_DIR = PROJECT_ROOT / "src" / "ml" / "models"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_synthetic_data(n_per_category: int = 500) -> pd.DataFrame:
    """
    Generate synthetic Norwegian transactions and label them.

    Returns:
        DataFrame with columns: [description, category, label_id, source]
    """
    print(f"[prepare] Generating {n_per_category} synthetic transactions per category...")
    transactions = generate_transactions(n_per_category=n_per_category)

    # All synthetic transactions are already labeled (category is set in generator)
    records = []
    for tx in transactions:
        records.append({
            "description": tx["description"],
            "category": tx["category"],
            "label_id": CATEGORY_TO_ID[tx["category"]],   # "groceries" → 0
            "source": "synthetic",
        })

    df = pd.DataFrame(records)
    print(f"[prepare] Synthetic: {len(df)} transactions across {df['category'].nunique()} categories")
    return df


def load_tink_data(client_id: str, client_secret: str, report_id: str = "", auth_code: str = "") -> pd.DataFrame:
    """
    Fetch real transactions from Tink and auto-label them.

    Two modes:
      report_id provided → Account Check flow (modern Tink API, uses browser-completed report)
      no report_id       → Legacy user/token flow (sandbox only, limited transactions)

    Returns:
        DataFrame with columns: [description, category, label_id, source]
        Only rows where auto_label() found a match (unlabeled rows excluded).
    """
    print("[prepare] Fetching Tink transactions...")

    if auth_code:
        from src.ml.data.tink_collector import exchange_code_for_transactions
        print(f"[prepare] Using authorization code: {auth_code[:8]}...")
        raw_transactions = exchange_code_for_transactions(
            client_id=client_id,
            client_secret=client_secret,
            authorization_code=auth_code,
        )
    elif report_id:
        from src.ml.data.tink_collector import get_account_check_transactions
        print(f"[prepare] Using Account Check report: {report_id[:8]}...")
        raw_transactions = get_account_check_transactions(
            client_id=client_id,
            client_secret=client_secret,
            report_id=report_id,
        )
    else:
        from src.ml.data.tink_collector import collect_training_data
        raw_transactions = collect_training_data(
            client_id=client_id,
            client_secret=client_secret,
        )

    print(f"[prepare] Tink returned {len(raw_transactions)} raw transactions")

    stats = get_coverage_stats([tx["description"] for tx in raw_transactions])
    print(f"[prepare] Rule coverage: {stats['coverage_pct']}% auto-labeled")

    records = []
    for tx in raw_transactions:
        result = auto_label(tx["description"])
        if result.category is not None:
            records.append({
                "description": tx["description"],
                "category": result.category,
                "label_id": CATEGORY_TO_ID[result.category],
                "source": "tink_sandbox",
            })

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["description", "category", "label_id", "source"]
    )
    print(f"[prepare] Tink labeled: {len(df)} transactions")
    return df


def load_gocardless_data(secret_id: str, secret_key: str) -> pd.DataFrame:
    """
    Fetch real transactions from GoCardless sandbox and auto-label them.

    Returns:
        DataFrame with columns: [description, category, label_id, source]
        Only rows where auto_label() found a match (unlabeled rows excluded).
    """
    from src.ml.data.psd2_collector import collect_training_data

    print("[prepare] Fetching GoCardless sandbox transactions...")
    raw_transactions = collect_training_data(
        secret_id=secret_id,
        secret_key=secret_key,
        use_sandbox=True,
    )

    print(f"[prepare] GoCardless returned {len(raw_transactions)} raw transactions")

    # Auto-label with our merchant rules
    # Transactions that don't match any rule are excluded (not "other")
    stats = get_coverage_stats([tx["description"] for tx in raw_transactions])
    print(f"[prepare] Rule coverage: {stats['coverage_pct']}% auto-labeled")

    records = []
    for tx in raw_transactions:
        result = auto_label(tx["description"])
        if result.category is not None:
            records.append({
                "description": tx["description"],
                "category": result.category,
                "label_id": CATEGORY_TO_ID[result.category],
                "source": "gocardless_sandbox",
            })

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["description", "category", "label_id", "source"]
    )
    print(f"[prepare] GoCardless labeled: {len(df)} transactions")
    return df


# ─── Dataset Splitting ────────────────────────────────────────────────────────

def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train / validation / test sets.

    WHY THIS SPLIT?
    ───────────────
    Imagine you're studying for an exam:
      - Train set   = your textbook (70%) — you study from this, model learns
      - Val set     = practice exams (15%) — you check how well you're learning
                      If practice exam score drops, you're cramming (overfitting) → stop early
      - Test set    = the REAL exam (15%) — you only take this ONCE at the end
                      Never look at test set during training — it's your honest score

    We use stratified splitting — each split has the same category distribution.
    Without stratification: by random chance, test might have 0 examples of "fuel".
    The model would score 100% on fuel in training but 0% in production.

    Args:
        df:          Full labeled DataFrame
        train_ratio: Fraction for training (default 70%)
        val_ratio:   Fraction for validation (default 15%)
        test_ratio:  Fraction for final test (default 15%)
        seed:        Random seed — same seed = same split every run (reproducibility)

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Step 1: Split off test set (15%)
    # stratify=df["category"] → ensures each category has same proportion in test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df["category"],   # ← crucial: balanced categories in each split
    )

    # Step 2: Split remaining (85%) into train + val
    # val_ratio relative to the remaining data = 0.15/0.85 ≈ 0.176
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=train_val_df["category"],
    )

    return train_df, val_df, test_df


# ─── TF-IDF Baseline ──────────────────────────────────────────────────────────

def train_tfidf_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Pipeline:
    """
    Train a TF-IDF + LogisticRegression baseline model.

    WHY BUILD A BASELINE BEFORE BERT?
    ───────────────────────────────────
    Always build the simplest possible model first.
    If TF-IDF gets 85% F1 in 10 seconds → BERT target is 92%+
    If TF-IDF gets 60% F1 → dataset has quality issues → fix data before BERT
    Never spend $6 on GPU training without knowing what "good enough" looks like.

    HOW TF-IDF WORKS:
    ─────────────────
    TF = Term Frequency: how often a word appears in THIS description
         "REMA 1000 OSLO REMA" → "REMA" appears 2/4 = 0.5 of words

    IDF = Inverse Document Frequency: how UNIQUE is this word across ALL descriptions?
         "REMA" appears in 20% of descriptions → log(1/0.20) = 1.61 (fairly unique)
         "NOK" appears in 90% of descriptions → log(1/0.90) = 0.10 (not useful)

    TF-IDF = TF × IDF:
         "REMA" in "REMA 1000 OSLO REMA": 0.5 × 1.61 = 0.805 (high score)
         "NOK" in any description: 0.5 × 0.10 = 0.05 (low score, almost ignored)

    Result: TF-IDF converts "REMA 1000 OSLO" into a vector like:
         [0.0, 0.805, 0.0, 0.0, 0.12, 0.0, ...]  ← 10,000-dim vector
    LogisticRegression then finds: "high REMA score + high 1000 score → groceries"

    WHAT IS A PIPELINE?
    ────────────────────
    sklearn Pipeline chains steps. When you call pipeline.fit(X, y):
      Step 1: TfidfVectorizer.fit_transform(X)  → learn vocabulary, convert to numbers
      Step 2: LogisticRegression.fit(X_tfidf, y)  → learn category weights
    When you call pipeline.predict(X):
      Step 1: TfidfVectorizer.transform(X)  → convert using LEARNED vocabulary
      Step 2: LogisticRegression.predict()  → output category
    One object, one .fit() call. Clean.

    Returns:
        Fitted sklearn Pipeline (TfidfVectorizer → LogisticRegression)
    """
    print("[prepare] Training TF-IDF baseline model...")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),   # Use single words AND two-word pairs
                                      # "CIRCLE K" = one feature (not "CIRCLE" + "K" separately)
                max_features=20_000,  # Keep top 20K most discriminative features
                sublinear_tf=True,    # Apply log(1+tf) — dampens very frequent words
                lowercase=False,      # Don't lowercase — "SAP" and "sap" differ in context
                analyzer="word",      # Split on whitespace (word-level tokens)
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,        # Enough iterations to converge
                C=1.0,                # Regularization strength (1.0 = balanced)
                                      # Low C = more regularization = less overfitting
                                      # High C = less regularization = might overfit
                solver="lbfgs",       # Limited-memory BFGS optimizer — fast for medium data
                                      # lbfgs handles multinomial multi-class natively;
                                      # multi_class param removed in sklearn 1.7+
                class_weight="balanced",   # Automatically handles class imbalance
                                           # Gives more weight to minority classes
                random_state=42,
            ),
        ),
    ])

    # Fit on training data only — never on val or test!
    pipeline.fit(train_df["description"].tolist(), train_df["label_id"].tolist())

    # Evaluate on validation set
    val_preds = pipeline.predict(val_df["description"].tolist())
    val_labels = val_df["label_id"].tolist()

    f1 = f1_score(val_labels, val_preds, average="weighted")
    print(f"[prepare] TF-IDF baseline val F1 (weighted): {f1:.4f}")
    print("[prepare] Per-class breakdown:")
    print(classification_report(
        val_labels,
        val_preds,
        target_names=CATEGORIES,
        digits=3,
    ))

    return pipeline


# ─── Parquet Output ───────────────────────────────────────────────────────────

def save_parquet_split(df: pd.DataFrame, path: Path, split_name: str) -> None:
    """
    Save a DataFrame split as a Parquet file.

    We reuse the same Parquet format as our Bronze data lake:
      - Columnar storage → fast column reads
      - Snappy compression → ~3x smaller than CSV
      - Schema enforcement → no type mismatches at training time

    Schema:
        description: string  — raw transaction description
        category:    string  — category name ("groceries", "fuel", ...)
        label_id:    int32   — integer label for neural network (0-6)
        source:      string  — "synthetic" or "gocardless_sandbox"
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # PyArrow schema — explicit types prevent silent type mismatches
    schema = pa.schema([
        ("description", pa.string()),
        ("category", pa.string()),
        ("label_id", pa.int32()),
        ("source", pa.string()),
    ])

    table = pa.Table.from_pandas(df[["description", "category", "label_id", "source"]], schema=schema)
    pq.write_table(table, path, compression="snappy")

    print(f"[prepare] Saved {split_name}: {len(df)} rows → {path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run(
    n_per_category: int = 500,
    use_gocardless: bool = False,
    gocardless_secret_id: str = "",
    gocardless_secret_key: str = "",
    use_tink: bool = False,
    tink_client_id: str = "",
    tink_client_secret: str = "",
    tink_report_id: str = "",
    tink_auth_code: str = "",
) -> None:
    """
    Full dataset preparation pipeline.

    Steps:
        1. Load synthetic data (always — fills bulk of training)
        2. Load Tink data (optional — real PSD2 sandbox transactions)
        3. Load GoCardless data (optional — real bank transactions)
        4. Combine sources and deduplicate
        5. Split 70/15/15
        6. Save Parquet splits
        7. Train TF-IDF baseline + save as joblib
        8. Print final statistics

    Args:
        n_per_category:          Synthetic transactions per category (500 recommended)
        use_tink:                Fetch Tink sandbox transactions?
        tink_client_id:          Tink app client ID
        tink_client_secret:      Tink app client secret
        use_gocardless:          Fetch real GoCardless sandbox transactions?
        gocardless_secret_id:    GoCardless credentials
        gocardless_secret_key:   GoCardless credentials
    """
    print("=" * 60)
    print("Finance Tracker ML — Dataset Preparation")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    dfs = [load_synthetic_data(n_per_category=n_per_category)]

    if use_tink and tink_client_id:
        try:
            tink_df = load_tink_data(tink_client_id, tink_client_secret, tink_report_id, tink_auth_code)
            if len(tink_df) > 0:
                dfs.append(tink_df)
        except Exception as e:
            print(f"[prepare] Tink failed (using synthetic only): {e}")

    if use_gocardless and gocardless_secret_id:
        try:
            gc_df = load_gocardless_data(gocardless_secret_id, gocardless_secret_key)
            if len(gc_df) > 0:
                dfs.append(gc_df)
        except Exception as e:
            print(f"[prepare] GoCardless failed (using synthetic only): {e}")

    # ── Step 2: Combine + deduplicate ─────────────────────────────────────────
    full_df = pd.concat(dfs, ignore_index=True)
    before = len(full_df)

    # Drop exact duplicate descriptions (same merchant, same label)
    full_df = full_df.drop_duplicates(subset=["description", "category"])
    print(f"[prepare] Deduplication: {before} → {len(full_df)} rows")

    # ── Step 3: Verify class balance ──────────────────────────────────────────
    print("[prepare] Category distribution:")
    for cat in CATEGORIES:
        count = (full_df["category"] == cat).sum()
        print(f"  {cat:12s}: {count:5d} examples")

    # ── Step 4: Split 70/15/15 ────────────────────────────────────────────────
    train_df, val_df, test_df = create_splits(full_df)
    print(f"\n[prepare] Split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    # ── Step 5: Save Parquet splits ────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet_split(train_df, DATA_DIR / "train.parquet", "train")
    save_parquet_split(val_df,   DATA_DIR / "val.parquet",   "val")
    save_parquet_split(test_df,  DATA_DIR / "test.parquet",  "test")

    # ── Step 6: Train TF-IDF baseline ─────────────────────────────────────────
    tfidf_pipeline = train_tfidf_baseline(train_df, val_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # joblib.dump = save Python object to disk as a binary file
    # joblib is better than pickle for numpy arrays (uses memory maps for large arrays)
    joblib.dump(
        tfidf_pipeline.named_steps["tfidf"],   # just the vectorizer
        MODELS_DIR / "tfidf_vectorizer.joblib",
    )
    joblib.dump(
        tfidf_pipeline.named_steps["clf"],     # just the classifier
        MODELS_DIR / "tfidf_classifier.joblib",
    )
    print(f"[prepare] TF-IDF models saved to {MODELS_DIR}")

    # ── Step 7: Final evaluation on test set ──────────────────────────────────
    test_preds = tfidf_pipeline.predict(test_df["description"].tolist())
    test_f1 = f1_score(test_df["label_id"].tolist(), test_preds, average="weighted")
    print(f"\n[prepare] TF-IDF TEST F1 (weighted): {test_f1:.4f}")
    print("[prepare] → If test F1 < 0.80: check data quality before BERT training")
    print("[prepare] → If test F1 > 0.90: BERT likely not needed for simple cases")
    print("[prepare] → Target: BERT achieves F1 > 0.95 (vs TF-IDF baseline)")

    print("\n[prepare] Done! Next steps:")
    print("  1. python -m src.ml.train.azure_ml_job   → submit BERT training to Azure ML")
    print("  2. python -m src.ml.train.export_onnx    → convert trained model to ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ML training dataset")
    parser.add_argument("--n-per-category", type=int, default=500)
    parser.add_argument("--use-tink", action="store_true")
    parser.add_argument("--tink-client-id", default=os.environ.get("TINK_CLIENT_ID", ""))
    parser.add_argument("--tink-client-secret", default=os.environ.get("TINK_CLIENT_SECRET", ""))
    parser.add_argument("--tink-report-id", default=os.environ.get("TINK_REPORT_ID", ""))
    parser.add_argument("--tink-auth-code", default="")
    parser.add_argument("--use-gocardless", action="store_true")
    parser.add_argument("--gc-secret-id", default=os.environ.get("GOCARDLESS_SECRET_ID", ""))
    parser.add_argument("--gc-secret-key", default=os.environ.get("GOCARDLESS_SECRET_KEY", ""))
    args = parser.parse_args()

    run(
        n_per_category=args.n_per_category,
        use_tink=args.use_tink,
        tink_client_id=args.tink_client_id,
        tink_client_secret=args.tink_client_secret,
        tink_report_id=args.tink_report_id,
        tink_auth_code=args.tink_auth_code,
        use_gocardless=args.use_gocardless,
        gocardless_secret_id=args.gc_secret_id,
        gocardless_secret_key=args.gc_secret_key,
    )
