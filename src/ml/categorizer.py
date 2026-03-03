"""
3-layer transaction categorizer for Norwegian financial transactions.

Architecture (cascade classifier):
  Layer 1 → Rules     (0ms)   — exact keyword lookup, 100% confidence
  Layer 2 → TF-IDF    (1ms)   — statistical ML, CPU, no model load
  Layer 3 → NB-BERT   (8ms)   — Norwegian BERT, ONNX int8, handles ambiguous cases

Each layer only activates if the previous layer is not confident enough.
Average latency: 0.80×0ms + 0.15×1ms + 0.05×8ms ≈ 0.55ms per transaction

WHY CASCADE?
──────────────
"REMA 1000 OSLO" → Layer 1 catches it instantly (REMA 1000 is in our rules)
"NYTT KAFÉ OG RESTAURANT AS" → Layer 1 fails, Layer 2 catches "KAFÉ" + "RESTAURANT"
"OSLO TEKNOLOGIPARTNERE AS" → Ambiguous. Layers 1+2 fail. Layer 3: context → "consulting"

This is the same architecture used by Vipps for transaction enrichment.
They call it "tiered classification" or "confidence-gated inference."

LAYER 3 (ONNX):
───────────────
Models are loaded ONCE when the class is instantiated (lazy loading optional).
ONNX Runtime session stays in memory for the lifetime of the FastAPI app.
No cold start after the first request — all subsequent inferences are 8ms.

CONFIDENCE THRESHOLD:
──────────────────────
Layer 2 (TF-IDF) returns probabilities via LogisticRegression.predict_proba().
If the highest probability > 0.85, we trust it (high confidence).
If 0.85 is too high: model falls through to Layer 3 too often.
If 0.85 is too low: model uses TF-IDF when BERT would have been better.
0.85 is tuned empirically — adjust based on your val set performance.

Usage:
    # Initialize once (loads models into memory)
    categorizer = TransactionCategorizer()

    # Categorize a single transaction
    result = categorizer.categorize("REMA 1000 MAJORSTUEN")
    # → CategorizeResult(category="groceries", confidence=1.0, layer="rules")

    # Categorize a batch
    results = categorizer.categorize_batch([
        "REMA 1000 OSLO",
        "ACCENTURE NORGE AS",
        "OSLO TEKNOLOGIPARTNERE AS",
    ])
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.ml.data.label_transactions import (
    CATEGORIES,
    ID_TO_CATEGORY,
    MERCHANT_RULES,
    auto_label,
)


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[2]
MODELS_DIR   = PROJECT_ROOT / "src" / "ml" / "models"

ONNX_MODEL_PATH       = MODELS_DIR / "categorizer_int8.onnx"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TFIDF_CLASSIFIER_PATH = MODELS_DIR / "tfidf_classifier.joblib"


# ─── Configuration ────────────────────────────────────────────────────────────

# TF-IDF confidence threshold: if max probability < this, fall through to BERT.
# 0.85 = "we need to be 85% sure before trusting TF-IDF"
# Lower threshold → more TF-IDF (faster but less accurate for edge cases)
# Higher threshold → more BERT (slower but more accurate)
TFIDF_CONFIDENCE_THRESHOLD = 0.85

# BERT max token length — must match the value used during training (64)
BERT_MAX_LENGTH = 64


# ─── Result Type ──────────────────────────────────────────────────────────────

@dataclass
class CategorizeResult:
    """
    Result from categorizing a single transaction.

    Returning a structured result (not just a string) enables:
      - Logging which layer handled each transaction (operational monitoring)
      - Confidence-based decisions (e.g., flag low-confidence for human review)
      - A/B testing: compare "layer 2 confidence 0.80" vs "layer 3 result"

    Fields:
        category:   One of CATEGORIES ("groceries", "software", etc.)
        confidence: 0.0 to 1.0 (rules=1.0, TF-IDF=model_probability, BERT=softmax)
        layer:      Which layer produced this result ("rules", "tfidf", "bert")
        latency_ms: How long categorization took (useful for monitoring SLAs)
    """
    category: str
    confidence: float
    layer: str                   # "rules", "tfidf", or "bert"
    latency_ms: float = 0.0
    matched_keyword: Optional[str] = None  # For rules layer — which keyword matched


# ─── Categorizer ──────────────────────────────────────────────────────────────

class TransactionCategorizer:
    """
    3-layer cascade categorizer for Norwegian transaction descriptions.

    Initialization loads all models into memory once.
    All categorize() calls after that are fast (no disk I/O).

    In FastAPI: instantiate ONCE at application startup (in lifespan).
    Don't create a new instance per request — model loading takes ~500ms.

    Example:
        categorizer = TransactionCategorizer()
        result = categorizer.categorize("SAP AG SOFTWARE LICENSE NOK 15000")
        # CategorizeResult(category="software", confidence=1.0, layer="rules")
    """

    def __init__(
        self,
        tfidf_vectorizer_path: Path = TFIDF_VECTORIZER_PATH,
        tfidf_classifier_path: Path = TFIDF_CLASSIFIER_PATH,
        onnx_model_path: Path = ONNX_MODEL_PATH,
    ):
        """
        Load all models into memory.

        Loading order (fastest to slowest):
          TF-IDF vectorizer:  ~10ms (sklearn, small joblib file)
          TF-IDF classifier:  ~5ms  (logistic regression weights)
          ONNX session:       ~500ms (first time — ONNX Runtime compiles the graph)
                              ~0ms  (subsequent uses — session stays in memory)

        Args:
            tfidf_vectorizer_path: Path to saved TfidfVectorizer joblib
            tfidf_classifier_path: Path to saved LogisticRegression joblib
            onnx_model_path:       Path to quantized ONNX model (.onnx)
        """
        self._tfidf_vectorizer = None
        self._tfidf_classifier = None
        self._onnx_session = None
        self._tokenizer = None

        # Load TF-IDF models (Layer 2)
        self._load_tfidf(tfidf_vectorizer_path, tfidf_classifier_path)

        # Load ONNX model (Layer 3)
        self._load_onnx(onnx_model_path)

        print("[categorizer] All models loaded. Ready for inference.")

    def _load_tfidf(self, vectorizer_path: Path, classifier_path: Path) -> None:
        """
        Load TF-IDF vectorizer and LogisticRegression classifier from disk.

        joblib.load = deserialize a Python object from a binary .joblib file.
        This is the reverse of joblib.dump() in prepare_dataset.py.
        Loading takes ~10ms. Object stays in memory for all future calls.
        """
        if not vectorizer_path.exists():
            print(f"[categorizer] WARNING: TF-IDF vectorizer not found at {vectorizer_path}")
            print("[categorizer] Layer 2 will be skipped. Run prepare_dataset.py first.")
            return

        if not classifier_path.exists():
            print(f"[categorizer] WARNING: TF-IDF classifier not found at {classifier_path}")
            return

        self._tfidf_vectorizer = joblib.load(vectorizer_path)
        self._tfidf_classifier = joblib.load(classifier_path)
        vocab_size = len(self._tfidf_vectorizer.vocabulary_)
        print(f"[categorizer] TF-IDF loaded: vocab_size={vocab_size:,} features")

    def _load_onnx(self, onnx_path: Path) -> None:
        """
        Load ONNX Runtime inference session and BERT tokenizer.

        ONNX Runtime session creation (~500ms on first call):
          - Loads the .onnx file
          - Compiles the computation graph for your CPU
          - Applies runtime optimizations (operator fusion, memory planning)
          After creation: all inferences are 8ms (no further compilation)

        InferenceSession providers:
          ["CPUExecutionProvider"] = run on CPU (always available)
          ["CUDAExecutionProvider", "CPUExecutionProvider"] = GPU with CPU fallback
          We use CPU — ONNX int8 is fast enough for our use case without GPU.
        """
        if not onnx_path.exists():
            print(f"[categorizer] WARNING: ONNX model not found at {onnx_path}")
            print("[categorizer] Layer 3 will be skipped. Run export_onnx.py first.")
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Suppress ONNX Runtime verbose logging during session creation
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # 0=verbose, 3=errors only

        self._onnx_session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )

        # Load the NB-BERT tokenizer (needed to convert text → token IDs)
        # The tokenizer is saved alongside the model in models/final_model/
        # It's small (~580KB) and loads in ~50ms
        tokenizer_path = onnx_path.parent / "final_model"
        if tokenizer_path.exists():
            self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            # Fallback: download from HuggingFace Hub (requires internet)
            self._tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base")

        print(f"[categorizer] ONNX model loaded: {onnx_path.name}")
        print(f"[categorizer] ONNX providers: {self._onnx_session.get_providers()}")

    # ── Layer 1: Rules ────────────────────────────────────────────────────────

    def _apply_rules(self, description: str) -> Optional[CategorizeResult]:
        """
        Layer 1: Exact keyword matching against MERCHANT_RULES dictionary.

        Time complexity: O(R×K) where R = categories (7), K = keywords per category (~15)
        Effective: O(105) string comparisons = essentially O(1)
        Latency: ~0.01ms (sub-microsecond per transaction)

        Returns None if no keyword matched → caller falls through to Layer 2.
        """
        result = auto_label(description)
        if result.category is not None:
            return CategorizeResult(
                category=result.category,
                confidence=1.0,              # Rules are 100% confident by definition
                layer="rules",
                matched_keyword=result.matched_keyword,
            )
        return None

    # ── Layer 2: TF-IDF ──────────────────────────────────────────────────────

    def _apply_tfidf(self, description: str) -> Optional[CategorizeResult]:
        """
        Layer 2: TF-IDF + LogisticRegression statistical classifier.

        HOW IT WORKS (repeated as promised):
          1. TfidfVectorizer converts description → 20,000-dim sparse vector
             Each dimension = TF-IDF score for one vocabulary term
             "REMA" → dimension 4521 has value 0.805 (high, unique to groceries)
             "NOK"  → dimension 1234 has value 0.02  (low, appears everywhere)

          2. LogisticRegression processes the sparse vector
             → outputs probabilities for each of 7 categories
             → [0.01, 0.02, 0.89, 0.02, 0.01, 0.01, 0.04]
             → category 2 (software) has 89% probability

          3. If max probability > 0.85 → we're confident → return result
             If max probability < 0.85 → uncertain → fall through to Layer 3

        Latency: ~1ms (matrix multiply on sparse vector, no GPU needed)

        Returns None if not confident enough → caller falls through to Layer 3.
        """
        if self._tfidf_vectorizer is None or self._tfidf_classifier is None:
            return None  # Layer 2 not available (model files not found)

        # Transform: description string → TF-IDF feature vector
        # [description] as list because vectorizer expects list of strings
        X = self._tfidf_vectorizer.transform([description])

        # Predict: TF-IDF vector → probability for each category
        # predict_proba returns array shape (1, 7) = one row, 7 probabilities
        probabilities = self._tfidf_classifier.predict_proba(X)[0]

        max_probability = probabilities.max()
        predicted_label_id = probabilities.argmax()

        if max_probability >= TFIDF_CONFIDENCE_THRESHOLD:
            return CategorizeResult(
                category=ID_TO_CATEGORY[predicted_label_id],
                confidence=float(max_probability),
                layer="tfidf",
            )

        # Not confident enough — let Layer 3 handle it
        return None

    # ── Layer 3: NB-BERT ONNX ────────────────────────────────────────────────

    def _apply_bert(self, description: str) -> CategorizeResult:
        """
        Layer 3: Norwegian BERT via ONNX Runtime.

        BERT's advantage over TF-IDF:
          TF-IDF: "OSLO TEKNOLOGIPARTNERE" → no keywords match any category
          BERT: understands "TEKNOLOGI" (technology) + "PARTNERE" (partners) →
                context → likely "consulting" (IT consulting firm)

        BERT processes the ENTIRE sequence with self-attention:
          "OSLO TEKNOLOGIPARTNERE AS" → each word attends to all other words
          "OSLO" sees "TEKNOLOGIPARTNERE" → geographically located tech firm
          "AS" (Norwegian for "Ltd") → it's a company → consulting/software

        This contextual understanding is why BERT costs 8ms instead of 1ms.

        Latency: ~8ms (ONNX int8 on CPU — vs 300ms for PyTorch fp32)

        This layer ALWAYS returns a result (fallback to "other" if truly unknown).
        """
        if self._onnx_session is None or self._tokenizer is None:
            # ONNX model not loaded — fallback to TF-IDF top prediction or "other"
            if self._tfidf_vectorizer is not None:
                X = self._tfidf_vectorizer.transform([description])
                probas = self._tfidf_classifier.predict_proba(X)[0]
                return CategorizeResult(
                    category=ID_TO_CATEGORY[probas.argmax()],
                    confidence=float(probas.max()),
                    layer="tfidf_fallback",
                )
            return CategorizeResult(category="other", confidence=0.0, layer="fallback")

        # Tokenize: text → token IDs
        # The tokenizer converts "OSLO TEKNOLOGIPARTNERE AS" into integer IDs
        # that NB-BERT was trained on. Norwegian-specific subword pieces.
        encoding = self._tokenizer(
            description,
            return_tensors="np",          # NumPy arrays — ONNX Runtime expects numpy
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LENGTH,
        )

        # Run ONNX inference
        # session.run(output_names, input_dict)
        # Returns list of numpy arrays — one per output_name
        outputs = self._onnx_session.run(
            ["logits"],
            {
                "input_ids":      encoding["input_ids"].astype(np.int64),
                "attention_mask": encoding["attention_mask"].astype(np.int64),
            }
        )

        # outputs[0] = logits array, shape: (1, 7)
        # [0] = first (only) item in batch
        logits = outputs[0][0]

        # Softmax: convert raw scores → probabilities that sum to 1.0
        # logits: [-2.3, 1.8, 0.4, -0.7, 1.2, -1.1, 0.9]
        # softmax: [0.02, 0.41, 0.10, 0.03, 0.22, 0.02, 0.16]
        # Numerical stability trick: subtract max before exp to avoid overflow
        exp_logits = np.exp(logits - logits.max())
        probabilities = exp_logits / exp_logits.sum()

        predicted_id = int(probabilities.argmax())
        confidence   = float(probabilities.max())

        return CategorizeResult(
            category=ID_TO_CATEGORY[predicted_id],
            confidence=confidence,
            layer="bert",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def categorize(self, description: str) -> CategorizeResult:
        """
        Categorize a single transaction description.

        Runs layers in order: rules → TF-IDF → BERT.
        Returns the first confident result.

        Args:
            description: Raw transaction description from bank or event store.
                         Example: "SAP AG SOFTWARE LICENSE NOK 15000.00"

        Returns:
            CategorizeResult with category, confidence, and which layer handled it.

        Example:
            >>> result = categorizer.categorize("REMA 1000 MAJORSTUEN")
            >>> result.category, result.confidence, result.layer
            ("groceries", 1.0, "rules")

            >>> result = categorizer.categorize("ACCENTURE NORGE AS BISTAND")
            >>> result.category, result.confidence, result.layer
            ("consulting", 1.0, "rules")

            >>> result = categorizer.categorize("OSLO TEKNOLOGIPARTNERE AS")
            >>> result.category, result.layer
            ("consulting", "bert")
        """
        start = time.perf_counter()

        # Layer 1: Rules (0ms)
        result = self._apply_rules(description)
        if result is not None:
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result

        # Layer 2: TF-IDF (1ms)
        result = self._apply_tfidf(description)
        if result is not None:
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result

        # Layer 3: NB-BERT ONNX (8ms)
        result = self._apply_bert(description)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def categorize_batch(self, descriptions: list[str]) -> list[CategorizeResult]:
        """
        Categorize a list of transaction descriptions.

        For Layer 1+2, we process individually (fast enough — O(1) per item).
        For Layer 3, we could batch BERT calls (more efficient on GPU).
        Current implementation: simple loop. Batching is a future optimization.

        Args:
            descriptions: List of transaction descriptions

        Returns:
            List of CategorizeResult, one per description, in same order.
        """
        return [self.categorize(desc) for desc in descriptions]

    def get_layer_stats(self, descriptions: list[str]) -> dict:
        """
        Analyze which layer handles what fraction of transactions.

        Run this on a sample of your production transactions to:
          - Verify rules cover expected 80% of traffic
          - Identify new merchants to add to MERCHANT_RULES
          - Monitor if BERT is overloaded (>20% = rules are outdated)

        Returns:
            {
                "total": 1000,
                "by_layer": {"rules": 820, "tfidf": 145, "bert": 35},
                "by_layer_pct": {"rules": 82.0, "tfidf": 14.5, "bert": 3.5},
                "avg_latency_ms": 0.42,
            }
        """
        results = self.categorize_batch(descriptions)
        by_layer: dict[str, int] = {"rules": 0, "tfidf": 0, "bert": 0}
        total_latency = 0.0

        for r in results:
            layer_key = r.layer.split("_")[0]  # "tfidf_fallback" → "tfidf"
            by_layer[layer_key] = by_layer.get(layer_key, 0) + 1
            total_latency += r.latency_ms

        n = len(results)
        return {
            "total": n,
            "by_layer": by_layer,
            "by_layer_pct": {k: round(v / n * 100, 1) for k, v in by_layer.items()},
            "avg_latency_ms": round(total_latency / n, 3),
        }
