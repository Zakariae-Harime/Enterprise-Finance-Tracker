import hashlib
import time
import aiohttp
from decimal import Decimal
from typing import List

from src.integrations.base import ERPAdapter
from src.integrations.models import ERPExpense, ERPInvoice, SyncResult

BASE_URL = "https://tripletex.no/v2"


class TripletexAdapter(ERPAdapter):

    def __init__(self, consumer_token: str, employee_token: str, company_id: int):
        self._consumer_token = consumer_token
        self._employee_token = employee_token
        self._company_id = company_id

    def _auth_header(self) -> dict:
        timestamp_ms = str(int(time.time() * 1000))
        raw = f"{self._consumer_token}:{self._employee_token}:{timestamp_ms}"
        checksum = hashlib.sha512(raw.encode()).hexdigest()
        return {"Tripletex-Token": f"{self._consumer_token}:{checksum}:{timestamp_ms}"}

    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        payload = {
            "date": expense.expense_date,
            "description": expense.description,
            "postings": [
                {
                    "description": expense.merchant_name,
                    "amountGross": float(expense.amount),
                    "currency": {"id": expense.currency},
                }
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/ledger/voucher",
                json=payload,
                headers={**self._auth_header(), "Content-Type": "application/json"},
            ) as resp:
                body = await resp.json()
                if resp.status == 201:
                    external_id = str(body.get("value", {}).get("id", ""))
                    return SyncResult(success=True, external_id=external_id)
                return SyncResult(success=False, error=str(body))

    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        params = {"dateFrom": since_date, "fields": "id,date,description,postings(*)"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/ledger/voucher",
                params=params,
                headers=self._auth_header(),
            ) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
                return [
                    ERPInvoice(
                        external_id=str(v["id"]),
                        amount=Decimal(str(v.get("postings", [{}])[0].get("amountGross", 0))),
                        currency=v.get("postings", [{}])[0].get("currency", {}).get("id", "NOK"),
                        description=v.get("description", ""),
                        vendor_name=v.get("postings", [{}])[0].get("description", ""),
                        invoice_date=v.get("date", since_date),
                    )
                    for v in body.get("values", [])
                ]
