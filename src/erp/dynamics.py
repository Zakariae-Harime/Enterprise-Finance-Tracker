import aiohttp
from decimal import Decimal
from typing import List

from src.integrations.base import ERPAdapter
from src.integrations.models import ERPExpense, ERPInvoice, SyncResult


class DynamicsAdapter(ERPAdapter):

    def __init__(self, tenant_id: str, client_id: str, client_secret: str, resource_url: str):
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._resource_url = resource_url.rstrip("/")

    async def _get_token(self) -> str:
        token_url = f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": f"{self._resource_url}/.default",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as resp:
                body = await resp.json()
                return body["access_token"]

    def _headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
        }

    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        token = await self._get_token()
        payload = {
            "msdyn_vendorname": expense.merchant_name,
            "msdyn_description": expense.description,
            "msdyn_invoicedate": expense.expense_date,
            "msdyn_totalamount": float(expense.amount),
            "transactioncurrencyid": expense.currency,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._resource_url}/api/data/v9.2/msdyn_vendorinvoiceheaders",
                json=payload,
                headers=self._headers(token),
            ) as resp:
                if resp.status == 204:
                    entity_id = resp.headers.get("OData-EntityId", "")
                    return SyncResult(success=True, external_id=entity_id)
                body = await resp.json()
                error_msg = body.get("error", {}).get("message", str(body))
                return SyncResult(success=False, error=error_msg)

    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        token = await self._get_token()
        params = {
            "$filter": f"msdyn_invoicedate ge {since_date}",
            "$select": "msdyn_vendorinvoiceid,msdyn_totalamount,msdyn_description,msdyn_vendorname,msdyn_invoicedate,msdyn_duedate",
            "$expand": "transactioncurrencyid($select=isocurrencycode)",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self._resource_url}/api/data/v9.2/msdyn_vendorinvoiceheaders",
                params=params,
                headers=self._headers(token),
            ) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
                return [
                    ERPInvoice(
                        external_id=inv["msdyn_vendorinvoiceid"],
                        amount=Decimal(str(inv.get("msdyn_totalamount", 0))),
                        currency=inv.get("transactioncurrencyid", {}).get("isocurrencycode", "NOK"),
                        description=inv.get("msdyn_description", ""),
                        vendor_name=inv.get("msdyn_vendorname", ""),
                        invoice_date=inv.get("msdyn_invoicedate", since_date),
                        due_date=inv.get("msdyn_duedate"),
                    )
                    for inv in body.get("value", [])
                ]
