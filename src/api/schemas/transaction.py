"""
  Transaction Schemas (DTOs - Data Transfer Objects)

  Pydantic models for:
    - Request validation (what client sends to create a transaction)
    - Response serialization (what API returns after creation/query)

  Follows same pattern as account.py schemas.
"""
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from uuid import UUID
from decimal import Decimal
from typing import Optional


class TransactionType(str, Enum):
    CREDIT = "credit"      # Money coming IN
    DEBIT = "debit"        # Money going OUT
    TRANSFER = "transfer"  # Between accounts


class ExpenseCategory(str, Enum):
    MEALS = "meals"
    SUPPLIES = "supplies"
    RENT = "rent"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"
    SOFTWARE = "software"
    HEALTHCARE = "healthcare"
    MARKETING = "marketing"
    CONSULTING = "consulting"
    TRAVEL = "travel"
    OTHER = "other"


class Currency(str, Enum):
    NOK = "NOK"
    DKK = "DKK"
    MAD = "MAD"
    SEK = "SEK"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"


class CreateTransactionRequest(BaseModel):
    """
    Request body for creating a new transaction.

    Example:
    {
        "amount": "250.00",
        "currency": "NOK",
        "transaction_type": "debit",
        "merchant_name": "REMA 1000 Grünerløkka",
        "description": "Weekly groceries",
        "category": "meals"
    }
    """
    amount: Decimal = Field(
        ...,
        gt=0,
        description="Transaction amount (must be positive)"
    )
    currency: Currency = Field(
        default=Currency.NOK,
        description="Transaction currency (default: NOK)"
    )
    transaction_type: TransactionType = Field(
        ...,
        description="Type of transaction: credit, debit, or transfer"
    )
    merchant_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the merchant (e.g., 'REMA 1000 Grünerløkka')"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional description of the transaction"
    )
    category: Optional[ExpenseCategory] = Field(
        default=None,
        description="Expense category (can be set later by ML or user)"
    )
    account_id: UUID = Field(
        ...,
        description="The account this transaction belongs to"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "amount": "250.00",
                "currency": "NOK",
                "transaction_type": "debit",
                "merchant_name": "REMA 1000 Grünerløkka",
                "description": "Weekly groceries",
                "category": "meals",
                "account_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class TransactionCreatedResponse(BaseModel):
    """Response after successfully creating a transaction."""
    transaction_id: UUID
    status: str = "created"
    message: str = "Transaction created successfully"


class TransactionResponse(BaseModel):
    """
    Full transaction details reconstructed from event replay.

    This is the read model — built by replaying all events
    for a transaction aggregate (Created, Categorized, Disputed, etc.).
    """
    transaction_id: UUID
    amount: Decimal
    currency: str
    transaction_type: str
    merchant_name: str
    description: Optional[str] = None
    category: Optional[str] = None
    is_disputed: bool = False
    dispute_reason: Optional[str] = None
    version: int
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    """Paginated list of transactions."""
    transactions: list[TransactionResponse]
    total: int
    limit: int
    offset: int
class CategorizeTransactionRequest(BaseModel):
    """
      Request body for categorizing a transaction.

      Three sources can trigger this:
        - "user": Manual categorization via UI
        - "ml_model": BERT classifier prediction
        - "rule": Rule-based (e.g., merchant "REMA" → meals)
    """
    category: ExpenseCategory = Field(
        ...,  # ... = required, no default
        description="Expense category to assign to the transaction"
    )
    categorized_by: str = Field(
        default="user",
        pattern="^(user|ml_model|rule)$", # Pydantic regex validation — rejects anything other than these three values
        description="Source of categorization (e.g., 'user', 'ml_model', 'rule')"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for ML-based categorization (0.0 to 1.0)"
    )
    class Config:
        json_schema_extra = {
            "example": {
                "category": "meals",
                "categorized_by": "ml_model",
                "confidence_score": 0.85
            }
        }
class DisputeTransactionRequest(BaseModel):
    """
      Request body for disputing a transaction.

      In banking (e.g., DNB, Nordea), disputed transactions
      are frozen while under investigation. The amount is held until
      the dispute is resolved via chargeback or upheld.
    """
    reason: str = Field(
        ...,
        min_length=10,  # Forces a meaningful reason — "no" won't pass
        max_length=1000,
        description="Reason for disputing the transaction"
    )
    class Config:
        json_schema_extra = {
            "example": {
                "reason": "Unauthorized charge - I did not make this purchase"
            }
        }
class TransactionUpdatedResponse(BaseModel):
    """Generic response after mutating an existing transaction."""
    transaction_id: UUID
    status: str
    message: str 
    version: int # new version number after the update