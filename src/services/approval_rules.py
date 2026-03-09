"""
ApprovalRuleEngine — amount-based escalation logic.

Reads can_approve_up_to from organization_members (passed in, not fetched here).
This keeps the rule engine pure Python — no DB, no async, fully unit-testable.

Rule:
  owner                → unlimited authority (can_approve_up_to = None)
  admin/finance        → can approve up to their can_approve_up_to limit
  employee             → cannot approve anything
"""
from decimal import Decimal
from typing import Optional


class InsufficientApprovalAuthorityError(Exception):
    """Raised when the approver's limit is below the expense amount."""
    pass


class ApprovalRuleEngine:
    # Roles that are allowed to approve at all
    APPROVER_ROLES = {"owner", "admin", "finance"}

    def check_can_approve(
        self,
        approver_role: str,
        can_approve_up_to: Optional[Decimal],
        expense_amount: Decimal,
    ) -> None:
        """
        Validates that the approver has authority over this expense amount.

        Raises InsufficientApprovalAuthorityError if not allowed.
        Returns None (implicit) if allowed.
        """
        if approver_role not in self.APPROVER_ROLES:
            raise InsufficientApprovalAuthorityError(
                f"Role '{approver_role}' is not authorized to approve expenses."
            )

        # owner has unlimited authority
        if approver_role == "owner":
            return

        # check the amount ceiling
        if can_approve_up_to is None or expense_amount > can_approve_up_to:
            raise InsufficientApprovalAuthorityError(
                f"Your approval authority ({can_approve_up_to} NOK) is below "
                f"the expense amount ({expense_amount} NOK). Escalate to a higher role."
            )
