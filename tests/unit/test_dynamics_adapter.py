import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from src.erp.dynamics import DynamicsAdapter
from src.integrations.models import ERPExpense


TENANT_ID     = "abc-tenant"
CLIENT_ID     = "my-client-id"
CLIENT_SECRET = "my-secret"
RESOURCE_URL  = "https://myorg.crm.dynamics.com"

EXPENSE = ERPExpense(
    expense_id=UUID("11111111-1111-1111-1111-111111111111"),
    amount=Decimal("4500.00"),
    currency="NOK",
    description="Cloud hosting Q1",
    merchant_name="AWS",
    expense_date="2026-03-12",
)


def _mock_resp(status: int, body: dict, headers: dict = None) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=body)
    resp.headers = headers or {}
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _mock_session(post_responses: list, get_response=None) -> MagicMock:
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(side_effect=post_responses)
    if get_response:
        session.get = MagicMock(return_value=get_response)
    return session


@pytest.mark.asyncio
async def test_push_expense_success():
    token_resp   = _mock_resp(200, {"access_token": "fake-token"})
    invoice_resp = _mock_resp(204, {}, headers={"OData-EntityId": "INV-999"})

    with patch("aiohttp.ClientSession", return_value=_mock_session([token_resp, invoice_resp])):
        adapter = DynamicsAdapter(TENANT_ID, CLIENT_ID, CLIENT_SECRET, RESOURCE_URL)
        result = await adapter.push_expense(EXPENSE)

    assert result.success is True
    assert result.external_id == "INV-999"


@pytest.mark.asyncio
async def test_push_expense_failure():
    token_resp   = _mock_resp(200, {"access_token": "fake-token"})
    invoice_resp = _mock_resp(400, {"error": {"message": "Invalid vendor"}})

    with patch("aiohttp.ClientSession", return_value=_mock_session([token_resp, invoice_resp])):
        adapter = DynamicsAdapter(TENANT_ID, CLIENT_ID, CLIENT_SECRET, RESOURCE_URL)
        result = await adapter.push_expense(EXPENSE)

    assert result.success is False
    assert "Invalid vendor" in result.error


@pytest.mark.asyncio
async def test_pull_invoices_returns_list():
    token_resp = _mock_resp(200, {"access_token": "fake-token"})
    invoices_body = {
        "value": [
            {
                "msdyn_vendorinvoiceid": "INV-001",
                "msdyn_totalamount": 9800.0,
                "transactioncurrencyid": {"isocurrencycode": "NOK"},
                "msdyn_description": "SaaS renewal",
                "msdyn_vendorname": "Salesforce",
                "msdyn_invoicedate": "2026-03-01",
                "msdyn_duedate": "2026-03-31",
            }
        ]
    }
    get_resp = _mock_resp(200, invoices_body)

    with patch("aiohttp.ClientSession", return_value=_mock_session([token_resp], get_response=get_resp)):
        adapter = DynamicsAdapter(TENANT_ID, CLIENT_ID, CLIENT_SECRET, RESOURCE_URL)
        invoices = await adapter.pull_invoices("2026-03-01")

    assert len(invoices) == 1
    assert invoices[0].external_id == "INV-001"
    assert invoices[0].amount == Decimal("9800.0")
