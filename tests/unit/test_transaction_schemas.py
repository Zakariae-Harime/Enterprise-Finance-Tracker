"""
Unit tests for transaction Pydantic schema validation.

Tests the system boundary — Pydantic rejects invalid input BEFORE
it ever reaches the domain logic or database. Each test verifies
that specific validation rules catch bad data with appropriate errors.

Pattern: Arrange-Act-Assert using pytest.raises(ValidationError)
"""
import pytest
from decimal import Decimal
from uuid import uuid4
from pydantic import ValidationError

from src.api.schemas.transaction import (
    CreateTransactionRequest,
    CategorizeTransactionRequest,
    DisputeTransactionRequest,
)


# A valid base payload — tests override specific fields to trigger validation
VALID_CREATE_PAYLOAD = {
    "amount": "250.00",
    "currency": "NOK",
    "transaction_type": "debit",
    "merchant_name": "REMA 1000 Grünerløkka",
    "account_id": str(uuid4()),
}


class TestCreateTransactionValidation:
    """Tests for CreateTransactionRequest — the API entry point for new transactions."""

    def test_valid_request_passes(self):
        """Sanity check: a well-formed request should create without errors."""
        req = CreateTransactionRequest(**VALID_CREATE_PAYLOAD)
        assert req.amount == Decimal("250.00")
        assert req.currency.value == "NOK"

    def test_rejects_negative_amount(self):
        """Negative amounts are invalid — transactions always use positive + type (debit/credit)."""
        # Arrange
        payload = {**VALID_CREATE_PAYLOAD, "amount": "-50.00"}

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            CreateTransactionRequest(**payload)
        assert "greater than 0" in str(exc_info.value).lower()

    def test_rejects_zero_amount(self):
        """Zero-value transactions have no financial meaning."""
        payload = {**VALID_CREATE_PAYLOAD, "amount": "0"}

        with pytest.raises(ValidationError):
            CreateTransactionRequest(**payload)

    def test_rejects_empty_merchant_name(self):
        """Merchant name is required for categorization and audit trail."""
        payload = {**VALID_CREATE_PAYLOAD, "merchant_name": ""}

        with pytest.raises(ValidationError):
            CreateTransactionRequest(**payload)

    def test_rejects_invalid_currency(self):
        """Only ISO 4217 codes from our Currency enum are accepted."""
        payload = {**VALID_CREATE_PAYLOAD, "currency": "BITCOIN"}

        with pytest.raises(ValidationError):
            CreateTransactionRequest(**payload)

    def test_rejects_invalid_transaction_type(self):
        """Only credit, debit, transfer are valid."""
        payload = {**VALID_CREATE_PAYLOAD, "transaction_type": "refund"}

        with pytest.raises(ValidationError):
            CreateTransactionRequest(**payload)

    def test_defaults_currency_to_nok(self):
        """Norwegian market default — if no currency specified, use NOK."""
        payload = {**VALID_CREATE_PAYLOAD}
        del payload["currency"]

        req = CreateTransactionRequest(**payload)
        assert req.currency.value == "NOK"

    def test_category_is_optional(self):
        """Category can be None — ML model or user sets it later."""
        req = CreateTransactionRequest(**VALID_CREATE_PAYLOAD)
        assert req.category is None


class TestCategorizeTransactionValidation:
    """Tests for CategorizeTransactionRequest — ML and user categorization input."""

    def test_valid_ml_categorization(self):
        req = CategorizeTransactionRequest(
            category="meals",
            categorized_by="ml_model",
            confidence_score=0.92,
        )
        assert req.confidence_score == 0.92

    def test_rejects_invalid_categorized_by(self):
        """Only 'user', 'ml_model', 'rule' are valid sources."""
        with pytest.raises(ValidationError):
            CategorizeTransactionRequest(
                category="meals",
                categorized_by="random_script",
            )

    def test_rejects_confidence_above_one(self):
        """Confidence score is a probability — max 1.0."""
        with pytest.raises(ValidationError):
            CategorizeTransactionRequest(
                category="meals",
                categorized_by="ml_model",
                confidence_score=1.5,
            )

    def test_rejects_negative_confidence(self):
        with pytest.raises(ValidationError):
            CategorizeTransactionRequest(
                category="meals",
                categorized_by="ml_model",
                confidence_score=-0.1,
            )

    def test_defaults_categorized_by_to_user(self):
        """If not specified, assume manual user categorization."""
        req = CategorizeTransactionRequest(category="meals")
        assert req.categorized_by == "user"


class TestDisputeTransactionValidation:
    """Tests for DisputeTransactionRequest — dispute reason quality gate."""

    def test_valid_dispute(self):
        req = DisputeTransactionRequest(
            reason="I did not authorize this charge on my account"
        )
        assert "authorize" in req.reason

    def test_rejects_short_reason(self):
        """Reasons under 10 chars are too vague for investigation."""
        with pytest.raises(ValidationError):
            DisputeTransactionRequest(reason="fraud")

    def test_rejects_empty_reason(self):
        with pytest.raises(ValidationError):
            DisputeTransactionRequest(reason="")
