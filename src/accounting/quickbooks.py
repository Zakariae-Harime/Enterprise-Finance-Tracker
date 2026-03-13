import aiohttp
from decimal import Decimal
from typing import List

from src.integrations.base import ERPAdapter
from src.integrations.models import ERPExpense, ERPInvoice, SyncResult

TOKEN_URL = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
BASE_URL  = "https://quickbooks.api.intuit.com/v3/company"


class QuickBooksAdapter(ERPAdapter):

    def __init__(self, client_id: str, client_secret: str, refresh_token: str, realm_id: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._realm_id = realm_id

    async def _get_token(self) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                },
                auth=aiohttp.BasicAuth(self._client_id, self._client_secret),
                headers={"Accept": "application/json"},
            ) as resp:
                body = await resp.json()
                return body["access_token"]

    def _headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        token = await self._get_token()
        payload = {
            "TxnDate": expense.expense_date,
            "EntityRef": {"name": expense.merchant_name},
            "CurrencyRef": {"value": expense.currency},
            "Line": [
                {
                    "Amount": float(expense.amount),
                    "DetailType": "AccountBasedExpenseLineDetail",
                    "Description": expense.description,
                    "AccountBasedExpenseLineDetail": {
                        "AccountRef": {"name": "Uncategorized Expense"}
                    },
                }
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/{self._realm_id}/purchase",
                json=payload,
                headers=self._headers(token),
                params={"minorversion": "65"},
            ) as resp:
                body = await resp.json()
                if resp.status == 200:
                    purchase = body.get("Purchase", {})
                    return SyncResult(success=True, external_id=str(purchase.get("Id", "")))
                return SyncResult(success=False, error=str(body))

    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        token = await self._get_token()
        query = f"SELECT * FROM Bill WHERE TxnDate >= '{since_date}'"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/{self._realm_id}/query",
                params={"query": query, "minorversion": "65"},
                headers=self._headers(token),
            ) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
                bills = body.get("QueryResponse", {}).get("Bill", [])
                return [
                    ERPInvoice(
                        external_id=str(bill["Id"]),
                        amount=Decimal(str(bill.get("TotalAmt", 0))),
                        currency=bill.get("CurrencyRef", {}).get("value", "NOK"),
                        description=bill.get("PrivateNote", ""),
                        vendor_name=bill.get("VendorRef", {}).get("name", ""),
                        invoice_date=bill.get("TxnDate", since_date),
                        due_date=bill.get("DueDate"),
                    )
                    for bill in bills
                ]
