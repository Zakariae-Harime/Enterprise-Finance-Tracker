import base64
import aiohttp
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional

from src.integrations.base import ERPAdapter
from src.integrations.models import ERPExpense, ERPInvoice, SyncResult

SANDBOX_URL    = "https://api.tripletex.io/v2"
PRODUCTION_URL = "https://tripletex.no/v2"

# Tripletex currency name → internal integer ID
# The API requires {"id": 1} for NOK, not the string "NOK"
CURRENCY_IDS = {
    "NOK": 1,
    "EUR": 2,
    "USD": 3,
    "GBP": 4,
    "SEK": 5,
    "DKK": 6,
}


class TripletexAdapter(ERPAdapter):
    """
    Tripletex v2 adapter — real API auth flow:

    Step 1: POST /token/session/:create?consumerToken=X&employeeToken=Y&expirationDate=YYYY-MM-DD
            → {"value": {"token": "SESSION_TOKEN"}}

    Step 2: All requests use HTTP Basic Auth:
            Authorization: Basic base64("0:SESSION_TOKEN")
            (username is always the literal "0" — represents the company context)

    Session tokens are valid until expirationDate (max ~1 day).
    We cache the token on the instance and reuse it across calls.
    """

    def __init__(
        self,
        consumer_token: str,
        employee_token: str,
        company_id: int = 0,
        use_sandbox: bool = True,
    ):
        self._consumer_token = consumer_token
        self._employee_token = employee_token
        self._company_id     = company_id
        self._base_url       = SANDBOX_URL if use_sandbox else PRODUCTION_URL
        self._session_token: Optional[str] = None

    async def _get_session_token(self) -> str:
        """
        Exchange consumer + employee tokens for a session token.
        Called once per adapter instance — result cached on self._session_token.
        """
        if self._session_token:
            return self._session_token

        expiration = (date.today() + timedelta(days=1)).isoformat()

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self._base_url}/token/session/:create",
                params={
                    "consumerToken":  self._consumer_token,
                    "employeeToken":  self._employee_token,
                    "expirationDate": expiration,
                },
            ) as resp:
                body = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(
                        f"Tripletex session creation failed ({resp.status}): {body}"
                    )
                self._session_token = body["value"]["token"]
                return self._session_token

    async def _auth_headers(self) -> dict:
        """
        Build HTTP Basic Auth header.
        Username is always "0" (Tripletex company context for API).
        Password is the session token.
        """
        token = await self._get_session_token()
        encoded = base64.b64encode(f"0:{token}".encode()).decode()
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type":  "application/json",
        }

    def _currency_id(self, currency_code: str) -> int:
        """Map ISO currency code to Tripletex internal integer ID."""
        return CURRENCY_IDS.get(currency_code.upper(), 1)  # default NOK

    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        """
        Push an approved expense as a ledger voucher.
        Tripletex endpoint: POST /ledger/voucher
        Returns 201 with {"value": {"id": <voucher_id>}} on success.
        """
        headers = await self._auth_headers()
        payload = {
            "date":        expense.expense_date,
            "description": expense.description,
            "voucherType": None,   # None = Tripletex picks default type
            "postings": [
                {
                    "description": expense.merchant_name,
                    "amountGross": float(expense.amount),
                    "amountVat":   0.0,
                    "amountNet":   float(expense.amount),
                    "currency":    {"id": self._currency_id(expense.currency)},
                    "vatType":     None,
                }
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/ledger/voucher",
                json=payload,
                headers=headers,
            ) as resp:
                body = await resp.json()
                if resp.status == 201:
                    external_id = str(body.get("value", {}).get("id", ""))
                    return SyncResult(success=True, external_id=external_id)
                error = body.get("message") or str(body)
                return SyncResult(success=False, error=error)

    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        """
        Pull ledger vouchers created after since_date.
        Tripletex endpoint: GET /ledger/voucher
        Response key is "values" (plural) — not "value".
        """
        headers = await self._auth_headers()
        params = {
            "dateFrom": since_date,
            "dateTo":   date.today().isoformat(),
            "fields":   "id,date,description,postings(amountGross,currency,description)",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self._base_url}/ledger/voucher",
                params=params,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
                return [
                    ERPInvoice(
                        external_id=str(v["id"]),
                        amount=Decimal(
                            str(v.get("postings", [{}])[0].get("amountGross", 0))
                        ),
                        currency=str(
                            v.get("postings", [{}])[0]
                             .get("currency", {})
                             .get("id", 1)
                        ),
                        description=v.get("description", ""),
                        vendor_name=v.get("postings", [{}])[0].get("description", ""),
                        invoice_date=v.get("date", since_date),
                    )
                    for v in body.get("values", [])
                ]
