import aiohttp
from decimal import Decimal
from typing import List, Optional

from src.integrations.base import ERPAdapter
from src.integrations.models import ERPExpense, ERPInvoice, SyncResult


class SAPAdapter(ERPAdapter):

    def __init__(self, base_url: str, company_db: str, username: str, password: str):
        self._base_url = base_url.rstrip("/")
        self._company_db = company_db
        self._username = username
        self._password = password

    async def _login(self) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/b1s/v1/Login",
                json={
                    "CompanyDB": self._company_db,
                    "UserName": self._username,
                    "Password": self._password,
                },
            ) as resp:
                body = await resp.json()
                return body["SessionId"]

    def _cookie(self, session_id: str) -> dict:
        return {"Cookie": f"B1SESSION={session_id}"}

    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        session_id = await self._login()
        payload = {
            "DocDate": expense.expense_date,
            "DocumentLines": [
                {
                    "ItemDescription": f"{expense.description} — {expense.merchant_name}",
                    "UnitPrice": float(expense.amount),
                    "Quantity": 1,
                    "Currency": expense.currency,
                }
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/b1s/v1/PurchaseInvoices",
                json=payload,
                headers=self._cookie(session_id),
            ) as resp:
                body = await resp.json()
                if resp.status == 201:
                    return SyncResult(success=True, external_id=str(body["DocEntry"]))
                error_msg = body.get("error", {}).get("message", {}).get("value", str(body))
                return SyncResult(success=False, error=error_msg)

    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        session_id = await self._login()
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self._base_url}/b1s/v1/PurchaseInvoices",
                params={"$filter": f"DocDate ge '{since_date}'"},
                headers=self._cookie(session_id),
            ) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
                return [
                    ERPInvoice(
                        external_id=str(inv["DocEntry"]),
                        amount=Decimal(str(inv.get("DocTotal", 0))),
                        currency=inv.get("DocCurrency", "NOK"),
                        description=inv.get("Comments", ""),
                        vendor_name=inv.get("CardName", ""),
                        invoice_date=inv.get("DocDate", since_date),
                        due_date=inv.get("DocDueDate"),
                    )
                    for inv in body.get("value", [])
                ]
