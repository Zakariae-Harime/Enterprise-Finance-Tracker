"""
Real Tripletex Sandbox Integration Test

Hits the actual Tripletex test environment (api.tripletex.io).
Requires real sandbox credentials in environment variables.

Setup:
  1. Go to https://tripletex.no → register a free test company
  2. Inside the company: Integrations → API → Generate Token
  3. Set environment variables:
       TRIPLETEX_CONSUMER_TOKEN=your-consumer-token
       TRIPLETEX_EMPLOYEE_TOKEN=your-employee-token

Run:
  pytest tests/integration/test_tripletex_sandbox.py -v -m sandbox

This test is excluded from CI — it requires live credentials and network.
"""
import os
import pytest
from decimal import Decimal
from uuid import uuid4

from src.accounting.tripletex import TripletexAdapter
from src.integrations.models import ERPExpense

pytestmark = pytest.mark.sandbox

CONSUMER_TOKEN = os.getenv("TRIPLETEX_CONSUMER_TOKEN")
EMPLOYEE_TOKEN = os.getenv("TRIPLETEX_EMPLOYEE_TOKEN")

EXPENSE = ERPExpense(
    expense_id=uuid4(),
    amount=Decimal("3750.00"),
    currency="NOK",
    description="Finance Tracker — Demo Consulting Invoice",
    merchant_name="Bouvet ASA",
    expense_date="2026-03-13",
    category="consulting_services",
)


@pytest.fixture
def adapter():
    if not CONSUMER_TOKEN or not EMPLOYEE_TOKEN:
        pytest.skip(
            "Set TRIPLETEX_CONSUMER_TOKEN and TRIPLETEX_EMPLOYEE_TOKEN to run sandbox tests"
        )
    return TripletexAdapter(
        consumer_token=CONSUMER_TOKEN,
        employee_token=EMPLOYEE_TOKEN,
        use_sandbox=True,
    )


@pytest.mark.asyncio
async def test_session_token_exchange(adapter):
    """Verify we can exchange tokens for a valid session."""
    token = await adapter._get_session_token()
    assert token is not None
    assert len(token) > 10
    print(f"\n  Session token obtained: {token[:20]}...")


@pytest.mark.asyncio
async def test_push_expense_to_sandbox(adapter):
    """
    Push a real expense to Tripletex sandbox and verify it is created.
    Stores the external_id so the next test can clean up.
    """
    result = await adapter.push_expense(EXPENSE)

    print(f"\n  Push result: success={result.success}")
    if result.success:
        print(f"  Voucher created: external_id={result.external_id}")
    else:
        print(f"  Error: {result.error}")

    assert result.success is True, f"Push failed: {result.error}"
    assert result.external_id != ""

    # Store for teardown
    test_push_expense_to_sandbox.created_id = result.external_id


@pytest.mark.asyncio
async def test_pull_invoices_from_sandbox(adapter):
    """
    Pull vouchers from the sandbox and verify our ACL mapping is correct.
    Since_date = today so we only get what we just created.
    """
    invoices = await adapter.pull_invoices("2026-03-13")

    print(f"\n  Pulled {len(invoices)} invoice(s) from sandbox")
    for inv in invoices:
        print(f"  - {inv.external_id}: {inv.amount} {inv.currency} from {inv.vendor_name}")

    assert isinstance(invoices, list)
    # May be empty if no vouchers exist yet — not a failure
    for inv in invoices:
        assert inv.external_id != ""
        assert inv.amount >= Decimal("0")


@pytest.mark.asyncio
async def test_full_roundtrip(adapter):
    """
    Full demo flow: push an expense → pull it back → verify it appears.
    This is the end-to-end proof that the adapter works with the live API.
    """
    # Push
    push_result = await adapter.push_expense(EXPENSE)
    assert push_result.success is True, f"Push failed: {push_result.error}"
    created_id = push_result.external_id
    print(f"\n  Created voucher: {created_id}")

    # Pull back — should find the voucher we just created
    invoices = await adapter.pull_invoices("2026-03-13")
    found = any(inv.external_id == created_id for inv in invoices)

    print(f"  Pulled {len(invoices)} vouchers. Found ours: {found}")
    assert found, (
        f"Voucher {created_id} not found in pull response. "
        f"Got: {[i.external_id for i in invoices]}"
    )
