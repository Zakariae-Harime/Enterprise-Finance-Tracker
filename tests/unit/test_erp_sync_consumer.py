import os
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

os.environ.setdefault("ENCRYPTION_KEY", "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.consumers.erp_sync_consumer import ERPSyncConsumer
from src.integrations.models import SyncResult
from src.services.credentials import encrypt_credentials


ORG_ID     = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
EXPENSE_ID = UUID("11111111-1111-1111-1111-111111111111")

EXPENSE_ROW = {
    "id": EXPENSE_ID,
    "amount": Decimal("7500.00"),
    "currency": "NOK",
    "description": "Deloitte consulting",
    "merchant_name": "Deloitte",
    "expense_date": "2026-03-12",
    "category": "consulting_services",
}

ENCRYPTED = encrypt_credentials({"consumer_token": "ct", "employee_token": "et", "company_id": 1})

INTEGRATION_ROW = {
    "id": UUID("22222222-2222-2222-2222-222222222222"),
    "provider": "tripletex",
    "encrypted_credentials": ENCRYPTED,
}

EVENT_DATA = {
    "aggregate_id": str(EXPENSE_ID),
    "org_id": str(ORG_ID),
    "event_type": "ExpenseApproved",
}


def _make_pool(expense_row, integration_rows):
    conn = MagicMock()
    conn.fetchrow  = AsyncMock(return_value=expense_row)
    conn.fetch     = AsyncMock(return_value=integration_rows)
    conn.execute   = AsyncMock(return_value=None)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__  = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=conn)
    return pool


@pytest.mark.asyncio
async def test_expense_approved_pushes_to_erp():
    pool = _make_pool(EXPENSE_ROW, [INTEGRATION_ROW])

    mock_adapter = MagicMock()
    mock_adapter.push_expense = AsyncMock(
        return_value=SyncResult(success=True, external_id="ERP-999")
    )

    with patch("src.consumers.erp_sync_consumer.get_adapter", return_value=lambda **kw: mock_adapter):
        consumer = ERPSyncConsumer(pool)
        await consumer.process_event("ExpenseApproved", EVENT_DATA)

    mock_adapter.push_expense.assert_called_once()
    call_arg = mock_adapter.push_expense.call_args[0][0]
    assert call_arg.expense_id == EXPENSE_ID
    assert call_arg.amount == Decimal("7500.00")
    assert call_arg.merchant_name == "Deloitte"


@pytest.mark.asyncio
async def test_expense_approved_no_integrations():
    pool = _make_pool(EXPENSE_ROW, [])

    mock_adapter = MagicMock()
    mock_adapter.push_expense = AsyncMock()

    with patch("src.consumers.erp_sync_consumer.get_adapter", return_value=lambda **kw: mock_adapter):
        consumer = ERPSyncConsumer(pool)
        await consumer.process_event("ExpenseApproved", EVENT_DATA)

    mock_adapter.push_expense.assert_not_called()


@pytest.mark.asyncio
async def test_expense_approved_expense_not_found():
    pool = _make_pool(expense_row=None, integration_rows=[INTEGRATION_ROW])

    mock_adapter = MagicMock()
    mock_adapter.push_expense = AsyncMock()

    with patch("src.consumers.erp_sync_consumer.get_adapter", return_value=lambda **kw: mock_adapter):
        consumer = ERPSyncConsumer(pool)
        await consumer.process_event("ExpenseApproved", EVENT_DATA)

    mock_adapter.push_expense.assert_not_called()


@pytest.mark.asyncio
async def test_unrelated_event_ignored():
    pool = _make_pool(EXPENSE_ROW, [INTEGRATION_ROW])

    mock_adapter = MagicMock()
    mock_adapter.push_expense = AsyncMock()

    with patch("src.consumers.erp_sync_consumer.get_adapter", return_value=lambda **kw: mock_adapter):
        consumer = ERPSyncConsumer(pool)
        await consumer.process_event("TransactionCreated", {"aggregate_id": str(EXPENSE_ID)})

    mock_adapter.push_expense.assert_not_called()
