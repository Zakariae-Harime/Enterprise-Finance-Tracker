"""
Unit tests for BudgetProjectionConsumer alert logic.

Tests _should_alert_threshold() and _should_alert_exceeded() — the pure
functions that determine when budget alerts fire.

Pure functions = no DB, no async, no mocking needed.
Pattern: Arrange-Act-Assert (same as test_account_projection_consumer.py)
"""
import pytest
from decimal import Decimal

from src.consumers.budget_projection_consumer import BudgetProjectionConsumer

class TestShouldAlertThreshold:
    
    """Tests for alert threshold logic — called on every BudgetCreated and BudgetUpdated event."""

    def test_below_threshold_no_alert(self):
 # 2000 spent of 3000 limit = 66% — below 80% threshold
        assert BudgetProjectionConsumer._should_alert_threshold(
            Decimal("2000"), Decimal("3000"), 0.8
        ) is False
    def test_at_threshold_fire_alert(self):
 # 2400 spent of 3000 limit = 80% — exactly at threshold → alert should fire
        assert BudgetProjectionConsumer._should_alert_threshold(
            Decimal("2400"), Decimal("3000"), 0.8
        ) is True

    def test_above_threshold_fire_alert(self):
        # 2900 spent of 3000 limit = 96% — well past 80% threshold
        assert BudgetProjectionConsumer._should_alert_threshold(
            Decimal("2900"), Decimal("3000"), 0.8
        ) is True

    def test_zero_limit_no_alert(self):
        # Guard against division by zero — limit=0 means no budget configured
        assert BudgetProjectionConsumer._should_alert_threshold(
            Decimal("100"), Decimal("0"), 0.8
        ) is False
class TestShouldAlertExceeded:
    """Tests for exceeded threshold logic — called on every BudgetCreated and BudgetUpdated event."""

    def test_below_limit_no_alert(self):
        # 2999.99 of 3000 — one øre under, no exceeded alert
        assert BudgetProjectionConsumer._should_alert_exceeded(
            Decimal("2999.99"), Decimal("3000")
        ) is False
    def test_at_limit_fire_alert(self):
        # 3000 of 3000 — exactly at limit → alert should fire
        assert BudgetProjectionConsumer._should_alert_exceeded(
            Decimal("3000"), Decimal("3000")
        ) is True
    def test_above_limit_fire_alert(self):
     # Overdraft scenario — spent more than budget
        assert BudgetProjectionConsumer._should_alert_exceeded(
            Decimal("3500"), Decimal("3000")
        ) is True
class TestBudgetScenarios:
    """
    Simulates a month of grocery spending against a 3000 NOK REMA budget.
    Mirrors how Sbanken tracks monthly category limits.
    """
    def test_grocery_budget_scenario(self):
        """
        Week 1: REMA 800 NOK  -> 800/3000 = 26%  -> no alert
        Week 2: REMA 900 NOK  -> 1700/3000 = 56% -> no alert
        Week 3: Kiwi 700 NOK  -> 2400/3000 = 80% -> threshold alert
        Week 4: Vinmonopolet  -> 2900/3000 = 96% -> still threshold (already alerted)
        Extra:  Meny 200 NOK  -> 3100/3000 > 100% -> exceeded alert
        """
        threshold = BudgetProjectionConsumer._should_alert_threshold
        exceeded = BudgetProjectionConsumer._should_alert_exceeded
        limit = Decimal("3000")

        assert threshold(Decimal("800"), limit, 0.8) is False
        assert threshold(Decimal("1700"), limit, 0.8) is False
        assert threshold(Decimal("2400"), limit, 0.8) is True
        assert threshold(Decimal("2900"), limit, 0.8) is True
        assert exceeded(Decimal("2900"), limit) is False
        assert exceeded(Decimal("3100"), limit) is True
 