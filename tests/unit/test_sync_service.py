import os
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

os.environ.setdefault("ENCRYPTION_KEY", "Rrcx7V1BgRERZsKT7cVGFOQwuxnCuGL4zfSfilo-bOw=")

from src.services.sync_service import SyncService
from src.integrations.models import ERPInvoice
from src.services.credentials import encrypt_credentials


ORG_ID           = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
INTEGRATION_ID   = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
SYNC_JOB_ID      = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")

FAKE_CREDENTIALS = {"consumer_token": "ct-xxx", "employee_token": "et-xxx", "company_id": 1}
ENCRYPTED_CREDS  = encrypt_credentials(FAKE_CREDENTIALS)

FAKE_INVOICES = [
    ERPInvoice(
        external_id="INV-1",
        amount=Decimal("5000.00"),
        currency="NOK",
        description="Consulting Q1",
        vendor_name="Deloitte",
        invoice_date="2026-03-01",
    )
]


def _make_pool(integration_row, sync_job_id=SYNC_JOB_ID):
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=integration_row)
    conn.execute  = AsyncMock(return_value=None)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__  = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=conn)
    return pool, conn


@pytest.mark.asyncio
async def test_run_sync_success():
    integration_row = {
        "id": INTEGRATION_ID,
        "provider": "tripletex",
        "encrypted_credentials": ENCRYPTED_CREDS,
        "last_synced_at": None,
    }
    pool, conn = _make_pool(integration_row)

    mock_adapter = MagicMock()
    mock_adapter.pull_invoices = AsyncMock(return_value=FAKE_INVOICES)

    with patch("src.services.sync_service.get_adapter", return_value=lambda **kw: mock_adapter):
        service = SyncService(pool)
        result = await service.run_sync(INTEGRATION_ID, ORG_ID)

    assert result.success is True
    assert result.records_synced == 1
    assert result.errors == []


@pytest.mark.asyncio
async def test_run_sync_integration_not_found():
    pool, _ = _make_pool(integration_row=None)

    service = SyncService(pool)
    result = await service.run_sync(INTEGRATION_ID, ORG_ID)

    assert result.success is False
    assert result.records_synced == 0
    assert "not found" in result.errors[0]


@pytest.mark.asyncio
async def test_run_sync_adapter_exception():
    integration_row = {
        "id": INTEGRATION_ID,
        "provider": "tripletex",
        "encrypted_credentials": ENCRYPTED_CREDS,
        "last_synced_at": None,
    }
    pool, conn = _make_pool(integration_row)

    mock_adapter = MagicMock()
    mock_adapter.pull_invoices = AsyncMock(side_effect=Exception("Connection refused"))

    with patch("src.services.sync_service.get_adapter", return_value=lambda **kw: mock_adapter):
        service = SyncService(pool)
        result = await service.run_sync(INTEGRATION_ID, ORG_ID)

    assert result.success is False
    assert "Connection refused" in result.errors[0]
