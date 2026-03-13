import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.accounting.tripletex import TripletexAdapter
from src.integrations.models import ERPExpense

pytestmark = pytest.mark.asyncio

EXPENSE = ERPExpense(
    expense_id=uuid4(),
    amount=Decimal("1500.00"),
    currency="NOK",
    description="Team lunch",
    merchant_name="Maaemo AS",
    expense_date="2026-03-10",
)


def _make_adapter() -> TripletexAdapter:
    return TripletexAdapter(
        consumer_token="test-consumer",
        employee_token="test-employee",
        company_id=12345,
        use_sandbox=True,
    )


def _mock_resp(status: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=body)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _mock_session(responses: list) -> MagicMock:
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    # put = session token exchange, post = push_expense
    call_count = {"n": 0}
    def side_effect(*_a, **_kw):
        i = call_count["n"]
        call_count["n"] += 1
        return responses[i]
    session.put  = MagicMock(side_effect=side_effect)
    session.post = MagicMock(side_effect=side_effect)
    session.get  = MagicMock(side_effect=side_effect)
    return session


async def test_push_expense_calls_correct_endpoint():
    token_resp   = _mock_resp(200, {"value": {"token": "sess-abc"}})
    invoice_resp = _mock_resp(201, {"value": {"id": 9001}})

    with patch("aiohttp.ClientSession", return_value=_mock_session([token_resp, invoice_resp])):
        adapter = _make_adapter()
        result  = await adapter.push_expense(EXPENSE)

    assert result.success is True
    assert result.external_id == "9001"


async def test_push_expense_uses_basic_auth():
    """Authorization header must be Basic base64("0:SESSION_TOKEN")."""
    import base64

    token_resp   = _mock_resp(200, {"value": {"token": "sess-abc"}})
    invoice_resp = _mock_resp(201, {"value": {"id": 9001}})

    session = _mock_session([token_resp, invoice_resp])

    with patch("aiohttp.ClientSession", return_value=session):
        adapter = _make_adapter()
        await adapter.push_expense(EXPENSE)

    headers = session.post.call_args[1].get("headers", {})
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("Basic ")
    decoded = base64.b64decode(headers["Authorization"][6:]).decode()
    assert decoded.startswith("0:")   # username is always "0"


async def test_push_expense_session_token_cached():
    """Session token endpoint (PUT) should only be called once per adapter instance."""
    token_resp    = _mock_resp(200, {"value": {"token": "sess-xyz"}})
    invoice_resp1 = _mock_resp(201, {"value": {"id": 1001}})
    invoice_resp2 = _mock_resp(201, {"value": {"id": 1002}})

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__  = AsyncMock(return_value=False)

    put_responses  = [token_resp]
    post_responses = [invoice_resp1, invoice_resp2]
    session.put  = MagicMock(side_effect=put_responses)
    session.post = MagicMock(side_effect=post_responses)

    with patch("aiohttp.ClientSession", return_value=session):
        adapter = _make_adapter()
        await adapter.push_expense(EXPENSE)
        await adapter.push_expense(EXPENSE)  # second call — no new token request

    assert session.put.call_count == 1   # token fetched once, cached second time


async def test_push_expense_returns_failure_on_error():
    token_resp   = _mock_resp(200, {"value": {"token": "sess-abc"}})
    invoice_resp = _mock_resp(400, {"message": "Invalid posting amount"})

    with patch("aiohttp.ClientSession", return_value=_mock_session([token_resp, invoice_resp])):
        adapter = _make_adapter()
        result  = await adapter.push_expense(EXPENSE)

    assert result.success is False
    assert "Invalid posting amount" in result.error


async def test_pull_invoices_returns_mapped_list():
    token_resp = _mock_resp(200, {"value": {"token": "sess-abc"}})

    vouchers_body = {
        "values": [
            {
                "id": 5001,
                "date": "2026-03-01",
                "description": "AWS invoice",
                "postings": [
                    {"amountGross": 9800.0, "currency": {"id": 1}, "description": "Amazon"}
                ],
            }
        ]
    }
    get_resp = _mock_resp(200, vouchers_body)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__  = AsyncMock(return_value=False)
    session.put = MagicMock(return_value=token_resp)
    session.get = MagicMock(return_value=get_resp)

    with patch("aiohttp.ClientSession", return_value=session):
        adapter  = _make_adapter()
        invoices = await adapter.pull_invoices("2026-03-01")

    assert len(invoices) == 1
    assert invoices[0].external_id == "5001"
    assert invoices[0].amount == Decimal("9800.0")
    assert invoices[0].vendor_name == "Amazon"


async def test_sandbox_url_used_by_default():
    """Adapter created with use_sandbox=True must hit api.tripletex.io."""
    token_resp   = _mock_resp(200, {"value": {"token": "sess-abc"}})
    invoice_resp = _mock_resp(201, {"value": {"id": 9001}})

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__  = AsyncMock(return_value=False)
    session.put  = MagicMock(return_value=token_resp)
    session.post = MagicMock(return_value=invoice_resp)

    with patch("aiohttp.ClientSession", return_value=session):
        adapter = _make_adapter()
        await adapter.push_expense(EXPENSE)

    token_url = session.put.call_args[0][0]
    assert "api.tripletex.io" in token_url

    voucher_url = session.post.call_args[0][0]
    assert "api.tripletex.io" in voucher_url
