"""
Shared data models for ERP integration — Data Transfer Objects (DTOs).

These are pure data containers. No logic, no HTTP, no DB.
Every adapter speaks this language — Tripletex, SAP, Dynamics all map to/from these.

Why DTOs?
  Without them, SyncService would need to know Tripletex JSON structure.
  With them, SyncService only knows ERPExpense/ERPInvoice — provider-agnostic.
"""
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from uuid import UUID


@dataclass
class ERPExpense:
    """Normalized expense to push into any ERP."""
    expense_id: UUID          # Finance Tracker internal ID
    amount: Decimal
    currency: str             # 'NOK', 'EUR', etc.
    description: str
    merchant_name: str
    expense_date: str         # ISO date 'YYYY-MM-DD'
    category: Optional[str] = None


@dataclass
class ERPInvoice:
    """Normalized invoice pulled from any ERP."""
    external_id: str          # ERP-side document ID (Tripletex voucher id, SAP DocEntry, etc.)
    amount: Decimal
    currency: str
    description: str
    vendor_name: str
    invoice_date: str         # ISO date 'YYYY-MM-DD'
    due_date: Optional[str] = None


@dataclass
class SyncResult:
    """Result of a single push or pull operation."""
    success: bool
    external_id: Optional[str] = None   # ERP doc ID on successful push
    error: Optional[str] = None
