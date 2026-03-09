"""
Unit tests for ApprovalRuleEngine.

Tests the amount-based escalation logic in isolation — no DB, no HTTP.
"""
import pytest
from decimal import Decimal
from src.services.approval_rules import ApprovalRuleEngine, InsufficientApprovalAuthorityError


def test_approver_with_enough_limit_can_approve():
    engine = ApprovalRuleEngine()
    # can_approve_up_to=10000, expense=5000 → allowed
    engine.check_can_approve(
        approver_role="finance",
        can_approve_up_to=Decimal("10000.00"),
        expense_amount=Decimal("5000.00"),
    )  # must NOT raise


def test_approver_with_insufficient_limit_raises():
    engine = ApprovalRuleEngine()
    with pytest.raises(InsufficientApprovalAuthorityError):
        engine.check_can_approve(
            approver_role="finance",
            can_approve_up_to=Decimal("5000.00"),
            expense_amount=Decimal("10000.00"),
        )


def test_owner_has_unlimited_authority():
    engine = ApprovalRuleEngine()
    # owner can_approve_up_to=None means unlimited
    engine.check_can_approve(
        approver_role="owner",
        can_approve_up_to=None,
        expense_amount=Decimal("999999.00"),
    )  # must NOT raise


def test_employee_cannot_approve():
    engine = ApprovalRuleEngine()
    with pytest.raises(InsufficientApprovalAuthorityError):
        engine.check_can_approve(
            approver_role="employee",
            can_approve_up_to=Decimal("0.00"),
            expense_amount=Decimal("100.00"),
        )
