"""
Unit tests for the 3-layer enterprise transaction categorizer.

Tests each layer independently — no trained models required.
These tests run on any machine (CI/CD, laptop) without GPU or model files.

Structure:
  TestLabelTransactions:     Layer 1 — pure Python merchant rule lookup
  TestSyntheticGenerator:    synthetic data generation and schema validation
  TestCategorizerWithMocks:  full categorizer with mocked TF-IDF + ONNX

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
      - Known enterprise vendors → correct category
      - Unknown vendors → None (no false positives)
      - Case insensitivity (lowercase input → still matches)
      - Norwegian vendor names (AS suffix, Norwegian characters) handled correctly
    """

    def test_salesforce_is_saas_software(self):
        """
        Salesforce is the world's largest CRM — every mid-to-large Norwegian
        company pays a monthly invoice. Should always resolve to saas_software.
        """
        result = auto_label("SALESFORCE.COM EMEA LTD SUBSCRIPTION")
        assert result.category == "saas_software"
        assert result.confidence == 1.0
        assert result.matched_keyword == "SALESFORCE"

    def test_github_is_saas_software(self):
        """GitHub Enterprise is the standard code platform in Norwegian tech companies."""
        result = auto_label("GITHUB ENTERPRISE SERVER LICENSE")
        assert result.category == "saas_software"

    def test_aws_is_cloud_infrastructure(self):
        """
        Amazon Web Services invoices are the most common cloud bill in Norwegian enterprises.
        AWS EMEA SARL is the European legal entity that appears on bank statements.
        """
        result = auto_label("AMAZON WEB SERVICES EMEA SARL")
        assert result.category == "cloud_infrastructure"

    def test_sap_is_saas_software(self):
        """
        SAP ERP is used by 80%+ of Norwegian enterprises (Equinor, DNB, Telenor all run SAP).
        SAP AG is the German legal entity — appears on vendor invoices exactly this way.
        """
        result = auto_label("SAP AG SOFTWARE LISENS NOK 25000")
        assert result.category == "saas_software"
        assert result.matched_keyword == "SAP AG"

    def test_deloitte_is_consulting_services(self):
        """
        Deloitte is Big4 — audit + advisory + tech consulting.
        Their invoices routinely appear in enterprise transaction data.
        """
        result = auto_label("DELOITTE AS RÅDGIVNING KONSULENTBISTAND 250000 NOK")
        assert result.category == "consulting_services"

    def test_sas_airlines_is_travel_expenses(self):
        """
        SAS Group is the dominant airline for Oslo-London, Oslo-Frankfurt business travel.
        Flight tickets on corporate cards should classify as travel_expenses.
        """
        result = auto_label("SAS GROUP BILLETT OSLO GARDERMOEN-LONDON")
        assert result.category == "travel_expenses"

    def test_zoom_is_telecommunications(self):
        """
        Zoom is classified as telecommunications — video conferencing is corporate comms.
        Same category as Telenor mobile plans: both are communication infrastructure.
        """
        result = auto_label("ZOOM VIDEO COMMUNICATIONS INC LICENSE")
        assert result.category == "telecommunications"

    def test_telenor_bedrift_is_telecommunications(self):
        """
        Telenor Bedrift is Telenor's enterprise division — mobile + broadband for companies.
        """
        result = auto_label("TELENOR BEDRIFT MOBILABONNEMENT FAKTURA")
        assert result.category == "telecommunications"

    def test_google_ads_is_marketing(self):
        """
        Google Ads is marketing spend, NOT cloud infrastructure.
        'GOOGLE CLOUD' → cloud_infrastructure, but 'GOOGLE ADS' → marketing_advertising.
        """
        result = auto_label("GOOGLE ADS IRELAND LIMITED NOK 45000")
        assert result.category == "marketing_advertising"
        assert result.matched_keyword == "GOOGLE ADS"

    def test_google_cloud_is_cloud_infrastructure(self):
        """
        Google Cloud must NOT match marketing — even though both start with 'GOOGLE'.
        'GOOGLE CLOUD' appears before 'GOOGLE ADS' in MERCHANT_RULES traversal order,
        and cloud_infrastructure is checked before marketing_advertising.
        """
        result = auto_label("GOOGLE CLOUD PLATFORM EMEA COMPUTE ENGINE")
        assert result.category == "cloud_infrastructure"

    def test_unknown_vendor_returns_none(self):
        """
        An unknown vendor must return None — NOT a guessed category.
        Rules never guess. Better to return None and let BERT decide.
        """
        result = auto_label("UKJENT TEKNOLOGIPARTNER AS")
        assert result.category is None
        assert result.confidence == 0.0
        assert result.matched_keyword is None

    def test_case_insensitive_matching(self):
        """
        Bank APIs return descriptions in various cases.
        "salesforce.com emea ltd" should match same as "SALESFORCE.COM EMEA LTD".
        """
        result = auto_label("salesforce.com emea ltd")
        assert result.category == "saas_software"

    def test_label_batch_returns_all_results(self):
        """
        label_batch() must return one result per input description.
        No descriptions should be silently dropped.
        """
        descriptions = [
            "SALESFORCE CRM ENTERPRISE",
            "UNKNOWN VENDOR AS",
            "AMAZON WEB SERVICES EMEA SARL",
        ]
        results = label_batch(descriptions)

        assert len(results) == 3
        assert results[0].category == "saas_software"
        assert results[1].category is None           # Unknown → None
        assert results[2].category == "cloud_infrastructure"

    def test_coverage_stats_structure(self):
        """
        get_coverage_stats() must return a dict with the required keys.
        Used by prepare_dataset.py to validate data quality before training.
        """
        descriptions = [
            "SALESFORCE.COM EMEA LTD",
            "AMAZON WEB SERVICES EMEA",
            "UKJENT LEVERANDOR AS",
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
        If these are inconsistent, the model maps outputs to wrong category names.

        Example of wrong mapping:
          CATEGORY_TO_ID["saas_software"] = 0
          ID_TO_CATEGORY[0] = "cloud_infrastructure"   ← WRONG
        """
        for cat in CATEGORIES:
            cat_id = CATEGORY_TO_ID[cat]
            assert ID_TO_CATEGORY[cat_id] == cat, (
                f"Mapping broken for '{cat}': "
                f"CATEGORY_TO_ID={cat_id} but ID_TO_CATEGORY[{cat_id}]='{ID_TO_CATEGORY[cat_id]}'"
            )


# ─── TestSyntheticGenerator ───────────────────────────────────────────────────

class TestSyntheticGenerator:
    """Tests for synthetic enterprise data generation."""

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
        """Different seeds → different datasets (locations vary for travel category)."""
        tx1 = generate_transactions(n_per_category=50, seed=42)
        tx2 = generate_transactions(n_per_category=50, seed=99)

        descriptions1 = set(tx["description"] for tx in tx1)
        descriptions2 = set(tx["description"] for tx in tx2)
        assert descriptions1 != descriptions2

    def test_amounts_are_positive(self):
        """All transaction amounts must be positive."""
        transactions = generate_transactions(n_per_category=20)
        for tx in transactions:
            assert tx["amount"] > 0, f"Negative amount in: {tx}"

    def test_currency_is_nok(self):
        """All synthetic transactions are denominated in Norwegian Krone."""
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

    Mock setup:
      TF-IDF mock: returns 0.90 confidence for "consulting_services" (index 2)
      BERT mock:   returns high logit for "travel_expenses" (index 3)
    """

    @pytest.fixture
    def categorizer(self):
        """
        Create a TransactionCategorizer with mocked TF-IDF + ONNX models.

        patch() temporarily replaces the _load_tfidf and _load_onnx methods
        with no-ops. After the test, the original methods are restored automatically.
        """
        from src.ml.categorizer import TransactionCategorizer

        with patch.object(TransactionCategorizer, "_load_tfidf", return_value=None):
            with patch.object(TransactionCategorizer, "_load_onnx", return_value=None):
                cat = TransactionCategorizer()

        # Inject mock TF-IDF vectorizer
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = MagicMock()  # sparse matrix mock
        cat._tfidf_vectorizer = mock_vectorizer

        # Inject mock TF-IDF classifier
        # Returns high confidence (0.90) for "consulting_services" (index 2)
        mock_classifier = MagicMock()
        probabilities = np.zeros(7)
        probabilities[2] = 0.90  # 90% confidence for "consulting_services" (index 2)
        mock_classifier.predict_proba.return_value = np.array([probabilities])
        cat._tfidf_classifier = mock_classifier

        # Inject mock ONNX session
        # Returns high logit for "travel_expenses" (index 3)
        mock_session = MagicMock()
        logits = np.zeros(7)
        logits[3] = 5.0   # High logit for "travel_expenses" (index 3)
        mock_session.run.return_value = [np.array([logits])]
        cat._onnx_session = mock_session

        # Inject mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids":      np.array([[101, 999, 102] + [0] * 61]),
            "attention_mask": np.array([[1, 1, 1] + [0] * 61]),
        }
        cat._tokenizer = mock_tokenizer

        return cat

    def test_known_vendor_handled_by_rules_layer(self, categorizer):
        """
        SALESFORCE is in MERCHANT_RULES → Layer 1 handles it.
        Layer 2 (TF-IDF) and Layer 3 (BERT) must NOT be called.
        """
        result = categorizer.categorize("SALESFORCE.COM EMEA LTD SUBSCRIPTION")

        assert result.category == "saas_software"
        assert result.confidence == 1.0
        assert result.layer == "rules"

        # Verify TF-IDF and ONNX were never called
        categorizer._tfidf_vectorizer.transform.assert_not_called()
        categorizer._onnx_session.run.assert_not_called()

    def test_high_confidence_tfidf_does_not_call_bert(self, categorizer):
        """
        When TF-IDF returns probability > 0.85, Layer 3 (BERT) must NOT run.
        Our mock classifier returns 0.90 for "consulting_services" — above threshold.
        """
        # Description with no match in MERCHANT_RULES — unknown vendor
        result = categorizer.categorize("OSLO TEKNISK LEVERANDOR AS")

        assert result.layer == "tfidf"
        assert result.category == "consulting_services"  # index 2 from mock
        assert result.confidence == 0.90

        # Verify BERT was not called (TF-IDF was confident enough)
        categorizer._onnx_session.run.assert_not_called()

    def test_low_confidence_tfidf_falls_through_to_bert(self, categorizer):
        """
        When TF-IDF returns probability < 0.85, Layer 3 (BERT) MUST run.
        We override the mock to return low confidence (0.60).
        """
        low_confidence_probas = np.zeros(7)
        low_confidence_probas[2] = 0.60  # Below TFIDF_CONFIDENCE_THRESHOLD
        categorizer._tfidf_classifier.predict_proba.return_value = np.array([low_confidence_probas])

        result = categorizer.categorize("UKJENT TEKNOLOGIPARTNER AS")

        assert result.layer == "bert"
        assert result.category == "travel_expenses"  # index 3 from BERT mock
        categorizer._onnx_session.run.assert_called_once()

    def test_categorize_result_has_all_required_fields(self, categorizer):
        """
        CategorizeResult must always have category, confidence, layer, latency_ms.
        Missing any field = bug in the result structure.
        """
        result = categorizer.categorize("SALESFORCE CRM ENTERPRISE")

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
        latency_ms must be >= 0 (we actually measured time).
        Not negative (clock error) and not > 1 second (something is very wrong).
        """
        result = categorizer.categorize("SALESFORCE CRM ENTERPRISE")

        assert result.latency_ms >= 0
        assert result.latency_ms < 1000

    def test_categorize_batch_returns_correct_count(self, categorizer):
        """
        categorize_batch(['a', 'b', 'c']) must return exactly 3 results.
        No results should be dropped or duplicated.
        """
        descriptions = [
            "SALESFORCE.COM EMEA LTD",
            "AMAZON WEB SERVICES EMEA SARL",
            "DELOITTE AS RÅDGIVNING",
        ]
        results = categorizer.categorize_batch(descriptions)

        assert len(results) == 3

    def test_categorize_batch_preserves_order(self, categorizer):
        """
        Results must be in the same order as inputs.
        Batch processing must not reorder results.
        """
        descriptions = [
            "SALESFORCE CRM ENTERPRISE",    # rules → saas_software
            "GITHUB ENTERPRISE SERVER",      # rules → saas_software
            "AMAZON WEB SERVICES EMEA SARL", # rules → cloud_infrastructure
        ]
        results = categorizer.categorize_batch(descriptions)

        assert results[0].category == "saas_software"
        assert results[1].category == "saas_software"
        assert results[2].category == "cloud_infrastructure"

    def test_confidence_between_zero_and_one(self, categorizer):
        """Confidence must always be a valid probability: 0.0 ≤ confidence ≤ 1.0."""
        descriptions = [
            "SALESFORCE CRM ENTERPRISE",    # rules (confidence=1.0)
            "SOME ENTERPRISE VENDOR AS",    # tfidf or bert
        ]
        for desc in descriptions:
            result = categorizer.categorize(desc)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Invalid confidence {result.confidence} for '{desc}'"
            )

    def test_category_always_in_valid_categories(self, categorizer):
        """Result category must always be one of the 7 known enterprise categories."""
        descriptions = [
            "SALESFORCE.COM EMEA LTD",
            "AMAZON WEB SERVICES EMEA",
            "DELOITTE AS RÅDGIVNING",
            "UKJENT LEVERANDOR AS",
        ]
        for desc in descriptions:
            result = categorizer.categorize(desc)
            if result.category != "other":
                assert result.category in CATEGORIES or result.category == "other", (
                    f"Unknown category '{result.category}' for '{desc}'"
                )
