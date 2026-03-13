import aiohttp
from decimal import Decimal
from typing import List

from src.integrations.base import ERPAdapter
from src.integrations.models import ERPExpense, ERPInvoice, SyncResult

TOKEN_URL = "https://identity.xero.com/connect/token"
BASE_URL  = "https://api.xero.com/api.xro/2.0"


class XeroAdapter(ERPAdapter):

    def __init__(self, client_id: str, client_secret: str, tenant_id: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id

    async def _get_token(self) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "scope": "accounting.transactions accounting.contacts",
                },
                auth=aiohttp.BasicAuth(self._client_id, self._client_secret),
            ) as resp:
                body = await resp.json()
                return body["access_token"]

    def _headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Xero-tenant-id": self._tenant_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        token = await self._get_token()
        payload = {
            "Type": "ACCPAY",
            "Contact": {"Name": expense.merchant_name},
            "Date": expense.expense_date,
            "CurrencyCode": expense.currency,
            "LineItems": [
                {
                    "Description": expense.description,
                    "UnitAmount": float(expense.amount),
                    "Quantity": 1,
                }
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/Invoices",
                json=payload,
                headers=self._headers(token),
            ) as resp:
                body = await resp.json()
                if resp.status == 200:
                    invoices = body.get("Invoices", [])
                    if invoices:
                        return SyncResult(success=True, external_id=invoices[0]["InvoiceID"])
                return SyncResult(success=False, error=str(body))

    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        token = await self._get_token()
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/Invoices",
                params={"where": f"Date >= DateTime({since_date.replace('-', ',')})", "Type": "ACCPAY"},
                headers=self._headers(token),
            ) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
                return [
                    ERPInvoice(
                        external_id=inv["InvoiceID"],
                        amount=Decimal(str(inv.get("AmountDue", 0))),
                        currency=inv.get("CurrencyCode", "NOK"),
                        description=inv.get("Reference", ""),
                        vendor_name=inv.get("Contact", {}).get("Name", ""),
                        invoice_date=inv.get("DateString", since_date)[:10],
                        due_date=inv.get("DueDateString", "")[:10] or None,
                    )
                    for inv in body.get("Invoices", [])
                ]
