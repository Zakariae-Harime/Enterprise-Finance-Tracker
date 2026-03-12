import pytest
from decimal import Decimal
from uuid import uuid4
from typing import List

from src.integrations.models import ERPExpense, ERPInvoice, SyncResult
from src.integrations.base import ERPAdapter

pytestmark = pytest.mark.asyncio
class FakeAdapter(ERPAdapter):
    """A fake ERP adapter for testing. It simulates pushing expenses and pulling invoices."""
    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        # Simulate a successful push by returning a SyncResult with success=True and a fake external_id
        return SyncResult(success=True, external_id="EXT-001")
    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        return []
async def test_push_expense_returns_sync_result():
    adapter = FakeAdapter()
    expense = ERPExpense(
        expense_id=uuid4(),
        amount=Decimal("1500.00"),
        currency="NOK",
        description="Team lunch",
        merchant_name="Maaemo AS",
        expense_date="2026-03-10",
    )
    result = await adapter.push_expense(expense)
    assert result.success is True
    assert result.external_id == "EXT-001"

async def test_pull_invoices_returns_list():
    adapter = FakeAdapter()
    invoices = await adapter.pull_invoices(since_date="2026-01-01")
    assert isinstance(invoices, list)
   
    

    

    