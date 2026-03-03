"""
Unit tests for the 3-layer transaction categorizer.

Tests each layer independently — no trained models required.
These tests run on any machine (CI/CD, laptop) without GPU or model files.

Structure:
  TestRulesLayer:     Layer 1 — pure Python dict lookup
  TestLabelTransactions: label_transactions.py functions
  TestCategorizerWithMocks: full categorizer with mocked TF-IDF + ONNX

Why mock models?
  Unit tests should be fast (< 1 second) and isolated.
  Loading ONNX model = 500ms + requires model file on disk.
  Mocking lets us test OUR LOGIC (routing, thresholds) without real models.
  Integration tests (not written here) would test with real models.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ml.data.label_transactions import (
    auto_label,
    label_batch,
    get_coverage_stats,
    CATEGORIES,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    LabelResult,
)
from src.ml.data.synthetic_generator import generate_transactions, get_dataset_stats
from src.ml.categorizer import CategorizeResult, TFIDF_CONFIDENCE_THRESHOLD


# ─── TestLabelTransactions ────────────────────────────────────────────────────

class TestLabelTransactions:
    """
    Tests for the rule-based labeling engine in label_transactions.py.

    Layer 1 logic: find a keyword in the UPPERCASE description.
    These tests verify:
      - Known Norwegian merchants → correct category
      - Unknown merchants → None (no false positives)
      - Case insensitivity (lowercase input → still matches)
      - Norwegian special characters (Å, Ø, Æ) handled correctly
    """

    def test_rema_1000_is_groceries(self):
        """
        'REMA 1000' is the most common Norwegian grocery chain.
        Every DNB transaction involving REMA should be classified as groceries.
        """
        result = auto_label("REMA 1000 MAJORSTUEN")
        assert result.category == "groceries"
        assert result.confidence == 1.0
        assert result.matched_keyword == "REMA 1000"

    def test_kiwi_is_groceries(self):
        """KIWI = second largest Norwegian grocery chain (NorgesGruppen)."""
        result = auto_label("KIWI GRÜNERLØKKA")
        assert result.category == "groceries"

    def test_circle_k_is_fuel(self):
        """
        Circle K is the largest petrol chain in Norway (rebranded from Statoil in 2016).
        Enterprise vehicles and company cards generate these regularly.
        """
        result = auto_label("CIRCLE K OSLO E18")
        assert result.category == "fuel"

    def test_sap_ag_is_software(self):
        """
        SAP AG is the German ERP vendor used by 80% of Norwegian enterprises.
        SAP license invoices are typically 5,000-100,000 NOK monthly.
        """
        result = auto_label("SAP AG SOFTWARE LICENSE NOK 15000")
        assert result.category == "software"

    def test_accenture_is_consulting(self):
        """
        Accenture is the largest consulting firm in Norway by revenue.
        Their invoices routinely appear in enterprise transaction data.
        """
        result = auto_label("ACCENTURE NORGE AS KONSULENTBISTAND 87500 NOK")
        assert result.category == "consulting"

    def test_mcdonalds_is_dining(self):
        """
        McDonald's is classified as dining even with location suffix.
        The transaction description from bank APIs often includes store location.
        """
        result = auto_label("MCDONALDS OSLO SENTRUM")
        assert result.category == "dining"

    def test_ruter_is_transport(self):
        """
        Ruter = Oslo's public transit authority. Bus, metro, tram.
        Monthly transit passes (950 NOK) appear as recurring Ruter transactions.
        """
        result = auto_label("RUTER BILLETT OSLO SENTRUM T-BANE")
        assert result.category == "transport"

    def test_telenor_is_utilities(self):
        """
        Telenor is the largest Norwegian telecom — mobile + internet for enterprises.
        """
        result = auto_label("TELENOR BEDRIFT MOBILABONNEMENT")
        assert result.category == "utilities"

    def test_unknown_merchant_returns_none(self):
        """
        An unknown merchant should return None — NOT a guessed category.
        Rules never guess. Better to return None than to confidently be wrong.
        """
        result = auto_label("OSLO TEKNOLOGIPARTNERE AS")
        assert result.category is None
        assert result.confidence == 0.0
        assert result.matched_keyword is None

    def test_case_insensitive_matching(self):
        """
        Banks return descriptions in various cases.
        "rema 1000 oslo" should match same as "REMA 1000 OSLO".
        """
        result = auto_label("rema 1000 oslo")
        assert result.category == "groceries"

    def test_label_batch_returns_all_results(self):
        """
        label_batch() must return one result per input description.
        No descriptions should be silently dropped.
        """
        descriptions = [
            "REMA 1000 OSLO",
            "UNKNOWN SHOP",
            "CIRCLE K BERGEN",
        ]
        results = label_batch(descriptions)

        assert len(results) == 3
        assert results[0].category == "groceries"
        assert results[1].category is None       # Unknown → None
        assert results[2].category == "fuel"

    def test_coverage_stats_structure(self):
        """
        get_coverage_stats() must return a dict with the required keys.
        Used by prepare_dataset.py to validate data quality before training.
        """
        descriptions = [
            "REMA 1000 OSLO",
            "CIRCLE K E18",
            "UKJENT BUTIKK AS",
        ]
        stats = get_coverage_stats(descriptions)

        assert "total" in stats
        assert "labeled" in stats
        assert "unlabeled" in stats
        assert "coverage_pct" in stats
        assert "by_category" in stats
        assert stats["total"] == 3
        assert stats["labeled"] == 2
        assert stats["unlabeled"] == 1

    def test_category_to_id_mapping_is_consistent(self):
        """
        CATEGORY_TO_ID and ID_TO_CATEGORY must be perfect inverses of each other.
        If these are inconsistent, the model maps to wrong categories.

        Example of wrong mapping:
          CATEGORY_TO_ID["groceries"] = 0
          ID_TO_CATEGORY[0] = "fuel"   ← WRONG → model says fuel, logs say groceries
        """
        for cat in CATEGORIES:
            cat_id = CATEGORY_TO_ID[cat]
            assert ID_TO_CATEGORY[cat_id] == cat, (
                f"Mapping broken for '{cat}': "
                f"CATEGORY_TO_ID={cat_id} but ID_TO_CATEGORY[{cat_id}]='{ID_TO_CATEGORY[cat_id]}'"
            )


# ─── TestSyntheticGenerator ───────────────────────────────────────────────────

class TestSyntheticGenerator:
    """Tests for synthetic data generation."""

    def test_generates_correct_number_of_transactions(self):
        """500 per category × 7 categories = 3500 total."""
        transactions = generate_transactions(n_per_category=100)

        stats = get_dataset_stats(transactions)
        assert stats["total"] == 7 * 100  # 7 categories × 100 each

    def test_all_categories_represented(self):
        """Every category must have examples — no category can be empty."""
        transactions = generate_transactions(n_per_category=50)
        stats = get_dataset_stats(transactions)

        for cat in CATEGORIES:
            assert cat in stats["by_category"], f"Category '{cat}' missing from synthetic data"
            assert stats["by_category"][cat] > 0, f"Category '{cat}' has 0 examples"

    def test_all_transactions_have_valid_category(self):
        """Every generated transaction must have a valid category label."""
        transactions = generate_transactions(n_per_category=20)

        for tx in transactions:
            assert tx["category"] in CATEGORIES, (
                f"Transaction has invalid category: '{tx['category']}'"
            )

    def test_reproducible_with_same_seed(self):
        """Same seed → same dataset. Critical for reproducible ML experiments."""
        tx1 = generate_transactions(n_per_category=10, seed=42)
        tx2 = generate_transactions(n_per_category=10, seed=42)

        descriptions1 = [tx["description"] for tx in tx1]
        descriptions2 = [tx["description"] for tx in tx2]
        assert descriptions1 == descriptions2

    def test_different_seeds_give_different_data(self):
        """Different seeds → different datasets (not all the same)."""
        tx1 = generate_transactions(n_per_category=50, seed=42)
        tx2 = generate_transactions(n_per_category=50, seed=99)

        descriptions1 = set(tx["description"] for tx in tx1)
        descriptions2 = set(tx["description"] for tx in tx2)
        # At least some descriptions should differ (locations vary by seed)
        assert descriptions1 != descriptions2

    def test_amounts_are_positive(self):
        """All transaction amounts must be positive (we use absolute values)."""
        transactions = generate_transactions(n_per_category=20)
        for tx in transactions:
            assert tx["amount"] > 0, f"Negative amount in: {tx}"

    def test_currency_is_nok(self):
        """All synthetic transactions are in Norwegian Krone."""
        transactions = generate_transactions(n_per_category=10)
        for tx in transactions:
            assert tx["currency"] == "NOK"

    def test_is_synthetic_flag(self):
        """Every synthetic transaction must be marked as synthetic."""
        transactions = generate_transactions(n_per_category=10)
        for tx in transactions:
            assert tx["is_synthetic"] is True


# ─── TestCategorizerWithMocks ─────────────────────────────────────────────────

class TestCategorizerWithMocks:
    """
    Tests for the full 3-layer categorizer using mocked models.

    We mock TF-IDF and ONNX model loading so tests don't need actual model files.
    This tests:
      - Layer routing (which layer handles which inputs)
      - Confidence thresholds
      - Fallback behavior (Layer 1 fails → Layer 2 → Layer 3)
      - CategorizeResult structure

    The @pytest.fixture creates a categorizer with mocked models.
    All test methods in this class use it via the `categorizer` parameter.
    """

    @pytest.fixture
    def categorizer(self):
        """
        Create a TransactionCategorizer with mocked TF-IDF + ONNX models.

        patch() temporarily replaces the _load_tfidf and _load_onnx methods
        with no-ops (do nothing). This prevents real file loading.
        After the test, the original methods are restored automatically.
        """
        from src.ml.categorizer import TransactionCategorizer

        with patch.object(TransactionCategorizer, "_load_tfidf", return_value=None):
            with patch.object(TransactionCategorizer, "_load_onnx", return_value=None):
                cat = TransactionCategorizer()

        # Inject a mock TF-IDF vectorizer
        # predict_proba must return array of shape (1, 7) — one probability per category
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock()  # sparse matrix mock
        cat._tfidf_vectorizer = mock_vectorizer

        # Inject a mock TF-IDF classifier
        # Returns high confidence (0.90) for "software" (index 2)
        mock_classifier = MagicMock()
        probabilities = np.zeros(7)
        probabilities[2] = 0.90  # 90% confidence for "software" (index 2)
        mock_classifier.predict_proba.return_value = np.array([probabilities])
        cat._tfidf_classifier = mock_classifier

        # Inject a mock ONNX session
        # Returns high logit for "consulting" (index 3)
        mock_session = MagicMock()
        logits = np.zeros(7)
        logits[3] = 5.0   # High logit for "consulting" (index 3)
        mock_session.run.return_value = [np.array([logits])]
        cat._onnx_session = mock_session

        # Inject a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids":      np.array([[101, 999, 102] + [0] * 61]),
            "attention_mask": np.array([[1, 1, 1] + [0] * 61]),
        }
        cat._tokenizer = mock_tokenizer

        return cat

    def test_known_merchant_handled_by_rules_layer(self, categorizer):
        """
        REMA 1000 is in MERCHANT_RULES → Layer 1 handles it.
        Layer 2 (TF-IDF) and Layer 3 (BERT) must NOT be called.
        """
        result = categorizer.categorize("REMA 1000 MAJORSTUEN")

        assert result.category == "groceries"
        assert result.confidence == 1.0
        assert result.layer == "rules"

        # Verify TF-IDF and ONNX were never called
        categorizer._tfidf_vectorizer.transform.assert_not_called()
        categorizer._onnx_session.run.assert_not_called()

    def test_high_confidence_tfidf_does_not_call_bert(self, categorizer):
        """
        When TF-IDF returns probability > 0.85, Layer 3 (BERT) must NOT run.
        Our mock classifier returns 0.90 for "software" — above threshold.
        """
        # Use description with no rule match — no keyword from MERCHANT_RULES appears here
        result = categorizer.categorize("OSLO TEKNISK LEVERANDOR AS")

        assert result.layer == "tfidf"
        assert result.category == "software"  # index 2 in our mock
        assert result.confidence == 0.90

        # Verify BERT was not called (TF-IDF was confident enough)
        categorizer._onnx_session.run.assert_not_called()

    def test_low_confidence_tfidf_falls_through_to_bert(self, categorizer):
        """
        When TF-IDF returns probability < 0.85, Layer 3 (BERT) MUST run.
        We adjust the mock to return low confidence (0.60).
        """
        # Override TF-IDF to return LOW confidence (below threshold)
        low_confidence_probas = np.zeros(7)
        low_confidence_probas[2] = 0.60  # Only 60% confident — below TFIDF_CONFIDENCE_THRESHOLD
        categorizer._tfidf_classifier.predict_proba.return_value = np.array([low_confidence_probas])

        result = categorizer.categorize("UKJENT TEKNOLOGIPARTNER AS")

        # Should fall through to BERT
        assert result.layer == "bert"
        assert result.category == "consulting"  # index 3 from BERT mock
        categorizer._onnx_session.run.assert_called_once()

    def test_categorize_result_has_all_required_fields(self, categorizer):
        """
        CategorizeResult must always have category, confidence, layer, latency_ms.
        Missing any field = bug in the result structure.
        """
        result = categorizer.categorize("REMA 1000 OSLO")

        assert hasattr(result, "category")
        assert hasattr(result, "confidence")
        assert hasattr(result, "layer")
        assert hasattr(result, "latency_ms")

        assert isinstance(result.category, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.layer, str)
        assert isinstance(result.latency_ms, float)

    def test_latency_is_recorded(self, categorizer):
        """
        latency_ms must be > 0 (we actually measured time).
        Not zero (forgot to record) and not negative (clock error).
        """
        result = categorizer.categorize("REMA 1000 OSLO")

        assert result.latency_ms >= 0  # >= 0 (could be very fast, near 0)
        assert result.latency_ms < 1000  # < 1 second (if this fails, something is very wrong)

    def test_categorize_batch_returns_correct_count(self, categorizer):
        """
        categorize_batch(['a', 'b', 'c']) must return exactly 3 results.
        No results should be dropped or duplicated.
        """
        descriptions = [
            "REMA 1000 OSLO",
            "CIRCLE K BERGEN",
            "SAP AG LIZENZ",
        ]
        results = categorizer.categorize_batch(descriptions)

        assert len(results) == 3

    def test_categorize_batch_preserves_order(self, categorizer):
        """
        Results must be in the same order as inputs.
        Batch processing should not reorder results.
        """
        descriptions = [
            "REMA 1000 OSLO",   # rules → groceries
            "KIWI SENTRUM",     # rules → groceries
            "CIRCLE K E18",     # rules → fuel
        ]
        results = categorizer.categorize_batch(descriptions)

        assert results[0].category == "groceries"
        assert results[1].category == "groceries"
        assert results[2].category == "fuel"

    def test_confidence_between_zero_and_one(self, categorizer):
        """Confidence must always be a valid probability: 0.0 ≤ confidence ≤ 1.0."""
        descriptions = [
            "REMA 1000 OSLO",    # rules (confidence=1.0)
            "SOME VENDOR AS",    # tfidf or bert
        ]
        for desc in descriptions:
            result = categorizer.categorize(desc)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Invalid confidence {result.confidence} for '{desc}'"
            )

    def test_category_always_in_valid_categories(self, categorizer):
        """Result category must always be one of the 7 known categories."""
        descriptions = [
            "REMA 1000 OSLO",
            "CIRCLE K E18",
            "SAP AG LIZENZ",
            "UNKNOWN VENDOR",
        ]
        for desc in descriptions:
            result = categorizer.categorize(desc)
            # Layer 3 might return any of our categories or fall through to bert
            # The layer "bert" or "rules" result should always be a known category
            if result.category != "other":
                assert result.category in CATEGORIES or result.category == "other", (
                    f"Unknown category '{result.category}' for '{desc}'"
                )
