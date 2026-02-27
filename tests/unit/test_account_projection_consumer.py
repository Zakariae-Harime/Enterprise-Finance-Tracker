"""
Unit tests for AccountProjectionConsumer balance calculation logic.

Tests _calculate_balance_delta() — the pure function that determines
how each transaction type affects the account balance.

Pure function = no database, no async, no mocking needed.
Same pattern as test_transaction_replay.py: Arrange-Act-Assert.
"""
import pytest
from decimal import Decimal

from src.consumers.account_projection_consumer import AccountProjectionConsumer


class TestCalculateBalanceDelta:
    """Tests for the core balance math — called on every TransactionCreated event."""

    def test_credit_returns_positive_delta(self):
        """
        CREDIT = money IN (e.g., salary, Vipps refund).
        Balance should INCREASE → positive delta.
        """
        # Arrange
        amount = Decimal("5000.00")

        # Act
        delta = AccountProjectionConsumer._calculate_balance_delta(amount, "credit")

        # Assert
        assert delta == Decimal("5000.00")

    def test_debit_returns_negative_delta(self):
        """
        DEBIT = money OUT (e.g., REMA groceries, Netflix subscription).
        Balance should DECREASE → negative delta.
        """
        amount = Decimal("250.00")

        delta = AccountProjectionConsumer._calculate_balance_delta(amount, "debit")

        assert delta == Decimal("-250.00")

    def test_transfer_returns_negative_delta(self):
        """
        TRANSFER = money leaving this account (current default: treat as debit).
        TODO(human) will decide final strategy.
        """
        amount = Decimal("1000.00")

        delta = AccountProjectionConsumer._calculate_balance_delta(amount, "transfer")

        assert delta == Decimal("-1000.00")

    def test_unknown_type_returns_zero(self):
        """
        Unknown types = no balance change.
        Safe default: don't guess, don't corrupt the balance.
        """
        amount = Decimal("100.00")

        delta = AccountProjectionConsumer._calculate_balance_delta(amount, "refund")

        assert delta == Decimal("0")

    def test_result_is_decimal_not_float(self):
        """
        Financial amounts MUST use Decimal — never float.

        Classic float bug:
            0.1 + 0.2 = 0.30000000000000004  ← float
            Decimal("0.1") + Decimal("0.2") = 0.3  ← correct

        Norwegian financial regulations require exact arithmetic.
        """
        amount = Decimal("0.10")

        delta = AccountProjectionConsumer._calculate_balance_delta(amount, "credit")

        assert isinstance(delta, Decimal)
        assert delta == Decimal("0.10")

    def test_large_amount_no_precision_loss(self):
        """Norwegian enterprise invoices can be in the millions."""
        amount = Decimal("9999999.99")

        delta = AccountProjectionConsumer._calculate_balance_delta(amount, "debit")

        assert delta == Decimal("-9999999.99")


class TestRunningBalanceScenarios:
    """
    Simulates sequences of transactions to verify running balance correctness.
    These tests mirror real account history at a Norwegian company.
    """

    def test_salary_deposit_then_expenses(self):
        """
        Common monthly scenario:
          1. Salary arrives (credit 50,000 NOK)
          2. Rent paid (debit 12,000 NOK)
          3. Groceries at REMA (debit 800 NOK)
          4. Vipps refund from friend (credit 200 NOK)

        Expected net: 50,000 - 12,000 - 800 + 200 = 37,400
        """
        calc = AccountProjectionConsumer._calculate_balance_delta
        starting_balance = Decimal("2000.00")

        transactions = [
            ("credit", "50000.00"),   # Salary
            ("debit",  "12000.00"),   # Rent
            ("debit",    "800.00"),   # REMA 1000
            ("credit",   "200.00"),   # Vipps refund
        ]

        balance = starting_balance
        for tx_type, amount_str in transactions:
            balance += calc(Decimal(amount_str), tx_type)

        assert balance == Decimal("39400.00")

    def test_balance_can_go_negative(self):
        """
        Overdraft is valid in Norwegian banking.
        The projection should allow negative balances — it's a ledger, not a gate.
        Overdraft rules are enforced at the bank level, not here.
        """
        calc = AccountProjectionConsumer._calculate_balance_delta
        balance = Decimal("100.00")

        balance += calc(Decimal("500.00"), "debit")

        assert balance == Decimal("-400.00")
