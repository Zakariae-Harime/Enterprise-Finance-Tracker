"""
GoCardless Bank Account Data API client (PSD2 Open Banking).

What is PSD2?
  EU Payment Services Directive 2 — a law that forces all European banks
  to provide API access to account transaction data. Every Norwegian bank
  (DNB, Sparebank, Nordea) must expose a standardized API.

What is GoCardless Bank Account Data (formerly Nordigen)?
  A company that aggregates all EU bank APIs into one simple REST API.
  Instead of integrating with DNB's API, Nordea's API, Sparebank's API separately,
  you integrate with GoCardless once — and get access to 2,500+ banks.

Why use it for ML training?
  Real transaction descriptions from real Norwegian banks.
  "REMA 1000 MAJORSTUEN 250.00 NOK" from a real DNB account.
  This is infinitely better training data than any Kaggle dataset.

GoCardless Sandbox:
  institution_id = "SANDBOXFINANCE_SFIN0000"
  No OAuth, no real bank needed.
  Returns pre-populated synthetic transactions in real GoCardless format.
  Perfect for testing and initial training data collection.

Sign up FREE at: https://bankaccountdata.gocardless.com/
  → Create account → Get your secret_id and secret_key
  → These are your credentials (like username/password for the API)

API Base URL: https://bankaccountdata.gocardless.com/api/v2/
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import uuid4

import httpx


# ─── Constants ────────────────────────────────────────────────────────────────

GOCARDLESS_BASE_URL = "https://bankaccountdata.gocardless.com/api/v2"

# Sandbox institution ID — use this for testing/training data.
# In production, use real institution IDs:
#   DNB:       "DNB_DNBANOKKXXX"
#   Sparebank: "SPAREBANK_1_SPNO2222"
#   Nordea:    "NORDEA_NDEANOKKXXX"
SANDBOX_INSTITUTION_ID = "SANDBOXFINANCE_SFIN0000"

# Sandbox pre-created account ID.
# In production flow, you get this AFTER the user authenticates via OAuth.
# For sandbox, GoCardless gives you a fixed account to query directly.
SANDBOX_ACCOUNT_ID = "3fa85f64-5717-4562-b3fc-2c963f66afa6"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class GoCardlessTransaction:
    """
    A single transaction from GoCardless API.

    GoCardless returns transactions in PSD2 format.
    We convert to our internal format for the training pipeline.

    Raw GoCardless format:
        {
            "transactionId": "2023123456789",
            "bookingDate": "2023-12-15",
            "transactionAmount": {"amount": "250.00", "currency": "NOK"},
            "creditorName": "REMA 1000 MAJORSTUEN",
            "remittanceInformationUnstructured": "REMA 1000 MAJORSTUEN  250 NOK",
        }
    """
    transaction_id: str
    description: str      # creditorName or remittanceInformationUnstructured
    amount: Decimal       # absolute value (we don't care about debit/credit direction here)
    currency: str
    booking_date: datetime
    raw: dict             # original GoCardless object (for debugging)


# ─── Client ───────────────────────────────────────────────────────────────────

class GoCardlessClient:
    """
    Client for GoCardless Bank Account Data API.

    Authentication flow:
        1. POST /token/new/ → get access_token (valid 24 hours)
        2. Use access_token in all subsequent requests

    Sandbox flow (no real bank needed):
        1. authenticate() → get token
        2. get_sandbox_transactions() → returns pre-populated transactions
        Done. No OAuth, no bank login needed.

    Production flow (real user's bank account):
        1. authenticate() → get token
        2. create_requisition() → create agreement with user's bank
        3. [User visits link, logs into their bank, grants permission]
        4. get_account_ids() → list of authorized account IDs
        5. get_transactions(account_id) → get real transactions

    Usage:
        client = GoCardlessClient(
            secret_id="your-secret-id",
            secret_key="your-secret-key",
        )
        client.authenticate()
        transactions = client.get_sandbox_transactions()
        # → List[GoCardlessTransaction] with real PSD2 format
    """

    def __init__(self, secret_id: str, secret_key: str):
        """
        Args:
            secret_id:  Your GoCardless API secret ID (get from dashboard)
            secret_key: Your GoCardless API secret key (get from dashboard)
        """
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._access_token: Optional[str] = None

    def authenticate(self) -> None:
        """
        Request an access token from GoCardless.

        POST /token/new/
        Body: {"secret_id": "...", "secret_key": "..."}
        Response: {"access": "eyJ0eXAiOiJKV1Qi...", "refresh": "...", "access_expires": 86400}

        The access token is a JWT (JSON Web Token):
          eyJhbGciOiJIUzI1NiJ9    ← header (base64)
          .eyJ1c2VyX2lkIjoiMTIzIn0   ← payload (base64) — contains expiry, user info
          .SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV  ← signature

        Token expires in 24 hours. We don't cache/refresh here for simplicity.
        Production systems should cache and only re-authenticate when token expires.
        """
        response = httpx.post(
            f"{GOCARDLESS_BASE_URL}/token/new/",
            json={
                "secret_id": self._secret_id,
                "secret_key": self._secret_key,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        self._access_token = data["access"]

    def _get_headers(self) -> dict:
        """Build authorization headers for all authenticated API calls."""
        if not self._access_token:
            raise RuntimeError("Call authenticate() before making API requests.")
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def get_sandbox_transactions(self) -> list[GoCardlessTransaction]:
        """
        Fetch transactions from the GoCardless sandbox account.

        Sandbox gives ~50 pre-populated transactions:
          REMA 1000 OSLO, Circle K Bergen, SAP AG Lizenz, etc.
        Same transactions every run — for reproducibility.

        Returns:
            List of GoCardlessTransaction objects

        API call: GET /accounts/{account_id}/transactions/
        Response:
            {
                "transactions": {
                    "booked": [   ← completed transactions
                        {
                            "transactionId": "...",
                            "bookingDate": "2024-01-15",
                            "transactionAmount": {"amount": "-250.00", "currency": "NOK"},
                            "creditorName": "REMA 1000 MAJORSTUEN",
                        },
                        ...
                    ],
                    "pending": []  ← not yet cleared (we ignore these)
                }
            }
        """
        response = httpx.get(
            f"{GOCARDLESS_BASE_URL}/accounts/{SANDBOX_ACCOUNT_ID}/transactions/",
            headers=self._get_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        # "booked" = completed transactions. "pending" = still processing.
        # We only use booked — pending transactions might still be reversed/cancelled.
        booked = data.get("transactions", {}).get("booked", [])
        return [self._parse_transaction(tx) for tx in booked]

    def get_transactions(self, account_id: str) -> list[GoCardlessTransaction]:
        """
        Fetch transactions from a real user account.

        Same API as sandbox, but with a real account_id obtained after
        the user has completed the OAuth bank login flow.

        Args:
            account_id: UUID of the authorized account (from get_account_ids())

        Returns:
            List of GoCardlessTransaction objects with real transaction data
        """
        response = httpx.get(
            f"{GOCARDLESS_BASE_URL}/accounts/{account_id}/transactions/",
            headers=self._get_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        booked = data.get("transactions", {}).get("booked", [])
        return [self._parse_transaction(tx) for tx in booked]

    def create_requisition(self, redirect_url: str) -> dict:
        """
        Create a bank authorization link for a real user.

        Production flow:
            1. Call this method → get back a "link" URL
            2. Redirect user to that URL → user logs into their bank
            3. Bank redirects user back to your redirect_url
            4. Call get_account_ids(requisition_id) to get their account IDs
            5. Call get_transactions(account_id) to get their transactions

        Args:
            redirect_url: Where to send the user after bank login
                          Example: "https://yourapp.com/psd2/callback"

        Returns:
            {
                "id": "requisition-uuid",
                "link": "https://ob.gocardless.com/ob-auth/start?id=...",
                "accounts": []  ← empty until user completes login
            }
        """
        response = httpx.post(
            f"{GOCARDLESS_BASE_URL}/requisitions/",
            headers=self._get_headers(),
            json={
                "redirect": redirect_url,
                "institution_id": SANDBOX_INSTITUTION_ID,
                "reference": str(uuid4()),   # unique per requisition
                "agreement": "",
                "user_language": "NO",        # Norwegian interface
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def get_account_ids(self, requisition_id: str) -> list[str]:
        """
        Get account IDs after user has completed bank OAuth flow.

        Args:
            requisition_id: The "id" from create_requisition() response

        Returns:
            List of account ID strings (UUIDs)
            Example: ["3fa85f64-5717-4562-b3fc-2c963f66afa6"]
        """
        response = httpx.get(
            f"{GOCARDLESS_BASE_URL}/requisitions/{requisition_id}/",
            headers=self._get_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("accounts", [])

    @staticmethod
    def _parse_transaction(raw: dict) -> GoCardlessTransaction:
        """
        Convert a raw GoCardless API response dict to our GoCardlessTransaction type.

        GoCardless can return the description in different fields depending on bank:
          - creditorName: merchant name (e.g., "REMA 1000 MAJORSTUEN")
          - remittanceInformationUnstructured: free-text description
          - additionalInformation: supplementary info

        We prefer creditorName (cleaner) → remittanceInformationUnstructured (fallback).

        Amount: GoCardless returns as string "-250.00" (negative = money leaving account).
        We store absolute value — direction doesn't matter for categorization.
        """
        # Extract the best available description
        description = (
            raw.get("creditorName")
            or raw.get("remittanceInformationUnstructured")
            or raw.get("additionalInformation")
            or "UNKNOWN"
        )

        # Amount: "-250.00" → Decimal("250.00")
        amount_str = raw.get("transactionAmount", {}).get("amount", "0")
        amount = abs(Decimal(amount_str))

        currency = raw.get("transactionAmount", {}).get("currency", "NOK")

        # Parse booking date: "2024-01-15" → datetime
        booking_date_str = raw.get("bookingDate", "2024-01-01")
        booking_date = datetime.strptime(booking_date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

        return GoCardlessTransaction(
            transaction_id=raw.get("transactionId", str(uuid4())),
            description=description,
            amount=amount,
            currency=currency,
            booking_date=booking_date,
            raw=raw,
        )


def collect_training_data(
    secret_id: str,
    secret_key: str,
    use_sandbox: bool = True,
    account_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    High-level function: connect to GoCardless and return transactions as dicts.

    This is the entry point for the data pipeline.
    Output format matches our event store schema so the same Bronze layer
    ingest code works for both GoCardless data and internal events.

    Args:
        secret_id:   GoCardless secret ID
        secret_key:  GoCardless secret key
        use_sandbox: If True, use sandbox account (no real bank needed)
        account_ids: If use_sandbox=False, list of real account IDs to fetch

    Returns:
        List of transaction dicts ready for Bronze layer upload and labeling.
        Format:
            {
                "event_id": "uuid",
                "description": "REMA 1000 MAJORSTUEN",
                "amount": Decimal("250.00"),
                "currency": "NOK",
                "created_at": datetime(...),
                "source": "gocardless",
            }
    """
    client = GoCardlessClient(secret_id=secret_id, secret_key=secret_key)
    client.authenticate()

    if use_sandbox:
        transactions = client.get_sandbox_transactions()
    else:
        if not account_ids:
            raise ValueError("Provide account_ids when use_sandbox=False")
        transactions = []
        for account_id in account_ids:
            transactions.extend(client.get_transactions(account_id))

    # Convert to our internal dict format
    return [
        {
            "event_id": tx.transaction_id,
            "description": tx.description,
            "amount": tx.amount,
            "currency": tx.currency,
            "created_at": tx.booking_date,
            "source": "gocardless_sandbox" if use_sandbox else "gocardless_production",
        }
        for tx in transactions
    ]
