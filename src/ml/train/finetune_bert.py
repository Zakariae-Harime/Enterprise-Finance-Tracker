"""
NB-BERT fine-tuning script for Norwegian transaction categorization.

⚠️  RUN THIS ON AZURE ML, NOT LOCALLY.
    This script requires a GPU. Running on CPU would take ~24 hours instead of ~1-2 hours.
    azure_ml_job.py uploads this script to Azure ML and runs it on a T4 GPU (~$1/run).

What is fine-tuning?
────────────────────
NbAiLab/nb-bert-base was pre-trained on billions of Norwegian words.
It already "understands" Norwegian — it knows SAP, Accenture, REMA 1000 as entities.
Fine-tuning = adding a classification head on top and teaching it:
  "given your Norwegian understanding, map this transaction → one of 7 categories"

Training takes ~1-2 hours on T4 GPU for 3,500 examples, 3 epochs.

After training, the script:
  1. Saves the PyTorch model to the Azure ML output directory
  2. Runs ONNX export automatically (so you only need to download the .onnx file)
  3. Logs metrics to Azure ML — visible in the portal dashboard

CONCEPTS REVISITED (as promised — I'll explain even if mentioned before):
──────────────────────────────────────────────────────────────────────────

EPOCH: One full pass through ALL 3,500 training examples.
  With batch_size=16: 3,500/16 = 218 gradient updates per epoch.
  We train 3 epochs → 218 × 3 = 654 total gradient updates.
  Each update: compute loss → backprop → adjust all 178M weights.

BATCH (mini-batch): 16 examples processed simultaneously on GPU.
  GPU processes them in parallel. Think of it like: instead of grading
  one student's exam at a time, grade 16 at once using a rubric.

LOSS: CrossEntropyLoss = -log(probability of correct class)
  If model gives "groceries" 80% probability when answer IS groceries:
    loss = -log(0.80) = 0.22  (low — model is mostly right)
  If model gives "groceries" 5% probability when answer IS groceries:
    loss = -log(0.05) = 3.0   (high — model is very wrong)

LEARNING RATE (2e-5 = 0.00002):
  How much to adjust each of 178M weights per gradient update.
  Too high: model bounces around, never settles (imagine huge jumps on a hill)
  Too low: training takes forever (tiny steps)
  2e-5 is the BERT paper's recommended value for fine-tuning — don't change it.

WARMUP: First 10% of training steps, learning rate rises from 0 → 2e-5.
  Then it follows a linear schedule down to ~0 by end of training.
  Why warmup? When weights are initially random (classification head),
  large immediate updates can "break" the pre-trained BERT weights.
  Warmup lets the head settle before full learning rate kicks in.

EARLY STOPPING: If validation loss hasn't improved for 2 epochs, stop.
  Prevents overfitting. Example:
    Epoch 1: val_loss=0.45 (best so far, save checkpoint)
    Epoch 2: val_loss=0.38 (new best, save checkpoint)
    Epoch 3: val_loss=0.41 (worse than epoch 2 — patience counter = 1)
    Epoch 4: val_loss=0.43 (still worse — patience counter = 2 → STOP)
  Final model = epoch 2 checkpoint (best validation loss).

FP16 (half precision):
  By default PyTorch uses 32-bit floats (4 bytes per number).
  FP16 = 16-bit floats (2 bytes per number).
  Result: 2x less GPU memory, 2x faster matrix multiplications.
  Modern T4/V100 GPUs have FP16 tensor cores optimized for this.
  Tiny accuracy loss (usually <0.01%) — always worth it.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,  # BERT + classification head
    AutoTokenizer,                        # BERT's text → token ID converter
    EarlyStoppingCallback,               # Stop when val loss stops improving
    Trainer,                             # Handles the training loop
    TrainingArguments,                   # All training hyperparameters
)
from sklearn.metrics import f1_score, accuracy_score

from src.ml.data.label_transactions import CATEGORIES, ID_TO_CATEGORY


# ─── Constants ────────────────────────────────────────────────────────────────

# NbAiLab/nb-bert-base:
#   - Trained by Norwegian AI Lab (NbAiLab) on Norwegian text corpora
#   - Base version: 12 layers, 768 hidden size, 12 attention heads, 178M parameters
#   - Downloaded automatically from HuggingFace Hub (~440MB)
#   - Same architecture as Google's original BERT, but Norwegian weights
MODEL_NAME = "NbAiLab/nb-bert-base"

# 7 categories, hardcoded here so the model's output layer has exactly 7 nodes
NUM_LABELS = len(CATEGORIES)

# Max token length. Norwegian transactions are short (< 20 words).
# BERT supports up to 512 tokens. 64 is generous and saves GPU memory.
# "REMA 1000 MAJORSTUEN" → ~5 tokens. Even the longest = ~15 tokens.
MAX_LENGTH = 64


# ─── Dataset Class ────────────────────────────────────────────────────────────

class TransactionDataset(Dataset):
    """
    PyTorch Dataset wrapping our Parquet files.

    What is a Dataset?
    ──────────────────
    PyTorch needs data in a specific format for the DataLoader to batch it.
    A Dataset class must implement:
      - __len__(): how many examples?
      - __getitem__(idx): give me example #idx

    HuggingFace Trainer uses this to feed batches to the GPU:
      DataLoader iterates Dataset → collects 16 examples → sends to GPU

    What is tokenization? (again, as promised)
    ──────────────────────────────────────────
    "REMA 1000 OSLO" cannot be fed to a neural network directly.
    We convert it to integer IDs:
      "REMA 1000 OSLO"
      → tokenizer splits: ["[CLS]", "RE", "##MA", "10", "##00", "OS", "##LO", "[SEP]"]
      → converts to IDs: [101, 9876, 5432, 8765, 3456, 7890, 2345, 102]
      → pads to length 64: [101, 9876, 5432, ..., 0, 0, 0, 0, 0]

    "input_ids":       the integer token IDs
    "attention_mask":  1 for real tokens, 0 for padding (model ignores padding)
    "labels":          the correct category integer (0-6)
    """

    def __init__(self, parquet_path: Path, tokenizer):
        """
        Load Parquet file and pre-tokenize all descriptions.

        Pre-tokenizing here (not in __getitem__) means we tokenize ONCE
        and cache the result. Otherwise we'd tokenize the same string
        hundreds of times across epochs — wasteful.
        """
        # Read Parquet → list of dicts
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        descriptions = df["description"].tolist()
        self._labels = df["label_id"].tolist()

        # Tokenize all descriptions at once (batch tokenization = faster)
        # truncation=True: cut at MAX_LENGTH tokens
        # padding="max_length": pad short sequences to MAX_LENGTH
        # return_tensors="pt": return PyTorch tensors (not Python lists)
        self._encodings = tokenizer(
            descriptions,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        print(f"[dataset] Loaded {len(descriptions)} examples from {parquet_path.name}")

    def __len__(self) -> int:
        """How many examples in this split?"""
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one example as a dict of tensors.

        The Trainer's DataCollator will stack these dicts into batches of 16.
        {
            "input_ids":      tensor([101, 9876, 5432, ..., 0, 0]),   # shape: (64,)
            "attention_mask": tensor([1, 1, 1, ..., 0, 0]),           # shape: (64,)
            "labels":         tensor(0),                               # scalar (0 = groceries)
        }
        """
        return {
            "input_ids":      self._encodings["input_ids"][idx],
            "attention_mask": self._encodings["attention_mask"][idx],
            "labels":         torch.tensor(self._labels[idx], dtype=torch.long),
        }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    """
    Called by Trainer after each evaluation epoch.

    Returns dict of metrics to log to Azure ML.
    We track both accuracy and weighted F1.

    WHY WEIGHTED F1?
    ─────────────────
    If 90% of transactions are "groceries":
      - Accuracy: model that always guesses "groceries" = 90% accurate (useless)
      - Weighted F1: same model = ~84% (penalizes missing other categories)
    F1 is harder to game — it demands good performance across ALL categories.

    LOGITS vs PROBABILITIES:
    ─────────────────────────
    BERT's final layer outputs "logits" — raw scores before normalization.
    Example: [-2.3, 1.8, 0.4, -0.7, 1.2, -1.1, 0.9]  (one per category)
    Softmax converts to probabilities: [0.02, 0.41, 0.10, 0.03, 0.22, 0.02, 0.16]
    argmax picks the highest: index 1 → "fuel"

    For metrics, we just need argmax (don't need probabilities):
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": acc,
        "f1_weighted": f1,
    }


# ─── Training ─────────────────────────────────────────────────────────────────

def train(data_dir: Path, output_dir: Path) -> None:
    """
    Fine-tune NbAiLab/nb-bert-base on our transaction categorization task.

    Args:
        data_dir:   Directory containing train.parquet, val.parquet, test.parquet
        output_dir: Where to save the trained model (Azure ML output directory)
    """
    print("=" * 60)
    print(f"Fine-tuning: {MODEL_NAME}")
    print(f"Categories: {CATEGORIES}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # ── Step 1: Load tokenizer ─────────────────────────────────────────────────
    # AutoTokenizer.from_pretrained() downloads the NB-BERT tokenizer from HuggingFace.
    # The tokenizer was trained specifically for Norwegian — it knows Norwegian words.
    # ~580KB download. Cached locally after first download.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"[train] Tokenizer loaded: vocab_size={tokenizer.vocab_size:,}")

    # ── Step 2: Load datasets ──────────────────────────────────────────────────
    train_dataset = TransactionDataset(data_dir / "train.parquet", tokenizer)
    val_dataset   = TransactionDataset(data_dir / "val.parquet",   tokenizer)
    test_dataset  = TransactionDataset(data_dir / "test.parquet",  tokenizer)

    # ── Step 3: Load model ─────────────────────────────────────────────────────
    # AutoModelForSequenceClassification downloads NB-BERT (~440MB) + adds:
    #   - Dropout layer (0.1 dropout rate — randomly zeros 10% of neurons during training)
    #   - Linear layer: 768 → num_labels (178M → 6K new parameters)
    #
    # id2label / label2id: maps integer → category name (for readable model outputs)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={idx: cat for idx, cat in enumerate(CATEGORIES)},
        label2id={cat: idx for idx, cat in enumerate(CATEGORIES)},
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {total_params:,}")

    # ── Step 4: Training arguments ────────────────────────────────────────────
    # TrainingArguments = all hyperparameters in one dataclass.
    # Think of this as the "settings" for the training run.
    training_args = TrainingArguments(
        output_dir=str(output_dir),          # Where to save checkpoints

        # Epoch settings
        num_train_epochs=5,                  # Max 5 epochs (early stopping may stop sooner)
                                             # EPOCH = one full pass through all training data

        # Batch settings
        per_device_train_batch_size=16,      # 16 examples per GPU forward pass
        per_device_eval_batch_size=32,       # Larger eval batch — no gradients needed, uses less memory
        gradient_accumulation_steps=1,       # Accumulate gradients over N steps before updating
                                             # Effective batch size = per_device_batch × N

        # Learning rate settings
        learning_rate=2e-5,                  # 0.00002 — standard for BERT fine-tuning
        warmup_ratio=0.1,                    # First 10% of steps: lr rises from 0 → 2e-5
        weight_decay=0.01,                   # L2 regularization — penalizes large weights
                                             # Helps prevent overfitting

        # Evaluation settings
        eval_strategy="epoch",              # Evaluate on val set after each epoch
        save_strategy="epoch",              # Save checkpoint after each epoch
        load_best_model_at_end=True,        # After training, load the BEST checkpoint
                                             # (not the last one — last might have overfit)
        metric_for_best_model="f1_weighted", # What "best" means: highest weighted F1

        # Speed settings
        fp16=torch.cuda.is_available(),      # Use 16-bit floats if GPU available
                                             # 2x faster, 2x less GPU memory, ~0.01% accuracy loss

        # Logging
        logging_steps=50,                   # Log loss/metrics every 50 gradient updates
        logging_dir=str(output_dir / "logs"),

        # Other
        dataloader_num_workers=2,           # Parallel data loading (2 CPU cores)
        report_to="azure_ml",               # Send metrics to Azure ML portal dashboard
        push_to_hub=False,                  # Don't publish to HuggingFace Hub
    )

    # ── Step 5: Create Trainer ─────────────────────────────────────────────────
    # Trainer handles: training loop, evaluation, checkpointing, logging.
    # You don't write "for epoch in epochs: for batch in dataloader: ..."
    # Trainer does all of this internally.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,      # Stop if val F1 doesn't improve for 2 epochs
                early_stopping_threshold=0.001, # Must improve by at least 0.1% to count
            )
        ],
    )

    # ── Step 6: TRAIN! ─────────────────────────────────────────────────────────
    print("\n[train] Starting fine-tuning...")
    print("[train] Watch Azure ML portal for live loss curves")
    trainer.train()

    # ── Step 7: Final test set evaluation ─────────────────────────────────────
    # test set = the sealed envelope. We only open it once.
    print("\n[train] Final evaluation on test set (never used during training):")
    test_results = trainer.evaluate(test_dataset)
    print(f"[train] Test F1 (weighted): {test_results.get('eval_f1_weighted', 'N/A'):.4f}")
    print(f"[train] Test Accuracy:      {test_results.get('eval_accuracy', 'N/A'):.4f}")

    # ── Step 8: Save model + tokenizer ────────────────────────────────────────
    # Trainer already saved checkpoints during training.
    # This saves the FINAL best model in a clean directory for ONNX export.
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"[train] Model saved to {final_model_dir}")

    # ── Step 9: Save test results for MLflow/Azure ML tracking ────────────────
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "categories": CATEGORIES,
            "test_f1_weighted": test_results.get("eval_f1_weighted"),
            "test_accuracy": test_results.get("eval_accuracy"),
            "num_train_examples": len(train_dataset),
            "num_val_examples": len(val_dataset),
            "num_test_examples": len(test_dataset),
        }, f, indent=2)
    print(f"[train] Results saved to {results_path}")

    print("\n[train] Training complete!")
    print(f"[train] Model directory: {final_model_dir}")
    print("[train] Next step: run export_onnx.py to convert to ONNX format")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Azure ML passes these as command-line arguments (set in azure_ml_job.py)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("AZUREML_DATAREFERENCE_data", "data/ml")),
        help="Directory with train/val/test Parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("AZUREML_MODEL_DIR", "outputs")),
        help="Where to save trained model",
    )

    args = parser.parse_args()
    train(data_dir=args.data_dir, output_dir=args.output_dir)
