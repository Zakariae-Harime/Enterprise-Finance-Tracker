from abc import ABC, abstractmethod
from typing import List

from src.integrations.models import ERPExpense, ERPInvoice, SyncResult


class ERPAdapter(ABC):
    """Abstract base class for ERP adapters. Each ERP (Tripletex, SAP, Dynamics) implements this."""

    @abstractmethod
    async def push_expense(self, expense: ERPExpense) -> SyncResult:
        """Push one approved expense to the ERP. Returns SyncResult — never raises."""
        ...

    @abstractmethod
    async def pull_invoices(self, since_date: str) -> List[ERPInvoice]:
        """Pull invoices created after since_date (ISO format 'YYYY-MM-DD')."""
        ...
