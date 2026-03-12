import aiohttp
from decimal import Decimal
from typing import List

from src.integrations.base import ERPAdapter
from src.integrations.models import ERPExpense, ERPInvoice, SyncResult

BASE_URL = "https://api.vismanet.no/v1"


class VismaAdapter(ERPAdapter):

    def __init__(self, client_id: str, client_secret: str, company_id: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._company_id = company_id

    async def _get_token(self) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://identity.vismaonline.com/connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "vismanet:api",
                },
            ) as resp:
                body = await resp.json()
                return body["access_token"]

    def _headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "ipp-company-id": self._company_id,
            "Content-Type": "application/json",
        }

    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        token = await self._get_token()
        payload = {
            "description": f"{expense.description} — {expense.merchant_name}",
            "date": expense.expense_date,
            "currencyCode": expense.currency,
            "grossAmount": float(expense.amount),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/expenseclaim",
                json=payload,
                headers=self._headers(token),
            ) as resp:
                body = await resp.json()
                if resp.status in (200, 201):
                    return SyncResult(success=True, external_id=str(body.get("claimId", "")))
                return SyncResult(success=False, error=str(body))

    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        token = await self._get_token()
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/supplierinvoice",
                params={"dateFrom": since_date},
                headers=self._headers(token),
            ) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
                return [
                    ERPInvoice(
                        external_id=str(inv.get("invoiceNumber", "")),
                        amount=Decimal(str(inv.get("grossAmount", 0))),
                        currency=inv.get("currencyCode", "NOK"),
                        description=inv.get("description", ""),
                        vendor_name=inv.get("supplierName", ""),
                        invoice_date=inv.get("invoiceDate", since_date),
                        due_date=inv.get("dueDate"),
                    )
                    for inv in body.get("data", [])
                ]
