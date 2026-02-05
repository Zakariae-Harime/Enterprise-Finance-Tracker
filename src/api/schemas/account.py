"""
  Account Schemas (DTOs - Data Transfer Objects)

  Pydantic models for:
    - Request validation (what client sends)
    - Response serialization (what API returns)

  Pydantic provides:
    - Automatic validation (rejects invalid data)
    - Type coercion (converts "123" to 123)
    - Auto-generates OpenAPI docs
    - Used by all major Norwegian tech companies
"""
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from uuid import UUID
from decimal import Decimal
from typing import Optional

class Currency(str, Enum):
    NOK = "NOK"  # Norwegian Krone
    DKK = "DKK"  # Danish Krone
    MAD = "MAD"  # Moroccan Dirham
    SEK = "SEK"  # Swedish Krona
    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    AUD = "AUD"  # Australian Dollar
    CAD = "CAD"  # Canadian Dollar
    CHF = "CHF"  # Swiss Franc
class AccountType(str, Enum):
    BUSINESS = "BUSINESS"
    CHECKING = "CHECKING"
    SAVINGS = "SAVINGS"
class AccountCreatingRequest(BaseModel):
    """
      Request body for creating a new account.

      Example:
      {
          "name": "Business Savings",
          "currency": "NOK",
          "account_type": "savings",
          "initial_balance": "10000.00"
      }
    """
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Name of the account (2-100 characters)",
        example=["Business Savings"]
    )
    currency: Currency= Field(
        default=Currency.NOK,
        description="Currency of the account (default: NOK)")
    account_type: AccountType = Field(
        default=AccountType.SAVINGS,
        description="Type of the account)")
    initial_balance: Optional[Decimal] = Field(
        default=Decimal("0.00"),
        ge=0, # greater than or equal to 0
        description="Initial balance for the account"
    )
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "name": "NOK SAVINGS",
                "currency": "NOK",
                "account_type": "SAVINGS",
                "initial_balance": "10000.00"
            }
        }
    class AccountResponse(BaseModel):
        """
        Response after creating/fetching an account.
        """
        account_id:UUID
        name:str
        created_at:datetime
        version:int
        balance:Decimal
        currency:Currency
        account_type:AccountType
        class Config:
            from_attributes = True  # Allow creating from ORM objects
    class AccountCreatedResponse(BaseModel):
        """
        Response after successfully creating an account.
        """
        account_id: UUID
        status: "created"
        message: str = "Account created successfully"