"""
Tink Open Banking PSD2 client for fetching sandbox transaction data.

What is Tink?
  Tink is a Swedish fintech (acquired by Visa in 2022) that aggregates European
  bank APIs into one unified API. Used by PayPal, BNP Paribas, and major
  Nordic banks to access customer account data via PSD2.

Why Tink over GoCardless for Norwegian market?
  - Open signup (GoCardless disabled new applications in 2024)
  - Swedish company with deep Nordic bank coverage
  - Used by Vipps, BankID, and major Norwegian fintechs
  - 3,400+ European banks including all major Norwegian ones

Tink OAuth 2.0 flows:
  1. Client Credentials — your backend authenticates itself (no user involved)
  2. Authorization Code — user logs into their bank (production only)

Sandbox flow (no real bank login needed):
  1. authenticate_as_client()  → client_token (admin scope)
  2. create_sandbox_user()     → creates test user with pre-loaded transactions
  3. get_user_token()          → user_token (scoped to one user's accounts only)
  4. get_transactions()        → sandbox transaction data

Sign up at: https://console.tink.com/
  → Create app → Get CLIENT_ID and CLIENT_SECRET
  → Enable scopes: accounts:read, transactions:read, authorization:grant

API Base URL: https://api.tink.com
"""
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import uuid4

import httpx


# ─── Constants ────────────────────────────────────────────────────────────────

TINK_BASE_URL = "https://api.tink.com"

# Sandbox user external ID — any string you choose.
# Using a fixed ID means same user every run → same sandbox transactions → reproducible training data.
SANDBOX_USER_EXTERNAL_ID = "finance-tracker-sandbox-user-v1"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class TinkTransaction:
    """
    A single transaction from Tink Data API v2.

    Tink's /data/v2/transactions response format:
        {
            "id": "8c73ab28bfde4e91ac...",
            "accountId": "a3f8c2...",
            "amount": {
                "value": {"unscaledValue": "-2500", "scale": "2"},
                "currencyCode": "SEK"
            },
            "dates": {"booked": "2024-01-15"},
            "descriptions": {
                "display": "Spotify AB",
                "original": "SPOTIFY AB 08-SEP-2024"
            },
            "status": "BOOKED",
        }

    Amounts use Tink's unscaled format to avoid floating-point errors:
        unscaledValue="-2500", scale="2" → -2500 / 10^2 = -25.00 SEK
    """
    transaction_id: str
    description: str      # display or original description
    amount: Decimal       # absolute value (direction doesn't matter for categorization)
    currency: str
    booking_date: datetime
    raw: dict             # original Tink API object (for debugging)


# ─── Client ───────────────────────────────────────────────────────────────────

class TinkClient:
    """
    Client for Tink Open Banking API (PSD2).

    Authentication flows:
      Client Credentials (your backend → Tink):
        POST /api/v1/oauth/token
        grant_type=client_credentials, scope=authorization:grant
        → Returns a token your backend uses to create users and grants

      Authorization Code (user token — after creating grant):
        POST /api/v1/oauth/token
        grant_type=authorization_code, code={from grant}
        → Returns token scoped to one specific user's accounts only

    Sandbox flow (no real bank login needed):
        1. authenticate_as_client()  → client_token
        2. create_sandbox_user()     → creates test user with pre-loaded transactions
        3. get_user_token()          → user_token (accounts:read, transactions:read)
        4. get_accounts()            → list of sandbox accounts
        5. get_transactions()        → sandbox transaction data

    Usage:
        client = TinkClient(client_id="...", client_secret="...")
        client.authenticate_as_client()
        client.create_sandbox_user()
        client.get_user_token()
        transactions = client.get_transactions()
    """

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._client_token: Optional[str] = None   # manages users/grants (admin)
        self._user_token: Optional[str] = None     # reads account data (user-scoped)

    # ── Authentication ─────────────────────────────────────────────────────

    def authenticate_as_client(self) -> None:
        """
        Get a client credentials token (machine-to-machine, no user involved).

        Scope 'authorization:grant' lets us:
          - Create sandbox users
          - Generate authorization grant codes for user tokens

        POST /api/v1/oauth/token
        Body: client_id, client_secret, grant_type=client_credentials, scope=authorization:grant
        Response: {"access_token": "eyJ...", "token_type": "bearer", "expires_in": 1799}
        """
        response = httpx.post(
            f"{TINK_BASE_URL}/api/v1/oauth/token",
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "client_credentials",
                "scope": "authorization:grant user:create account-verification-reports:read",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        self._client_token = response.json()["access_token"]

    def _client_headers(self) -> dict:
        """Build authorization headers using the client (admin) token."""
        if not self._client_token:
            raise RuntimeError("Call authenticate_as_client() first.")
        return {"Authorization": f"Bearer {self._client_token}"}

    def _user_headers(self) -> dict:
        """Build authorization headers using the user-scoped token."""
        if not self._user_token:
            raise RuntimeError("Call get_user_token() first.")
        return {"Authorization": f"Bearer {self._user_token}"}

    # ── Sandbox User Management ────────────────────────────────────────────

    def create_sandbox_user(self, external_user_id: str = SANDBOX_USER_EXTERNAL_ID) -> None:
        """
        Create a sandbox user in Tink's system.

        Tink pre-populates sandbox users with realistic test transactions.
        Using a fixed external_user_id means same user every run.

        If user already exists (409 Conflict), we silently continue —
        idempotent design means this is safe to call multiple times.

        POST /api/v1/user/create
        Headers: Authorization: Bearer {client_token}
        Body: {"external_user_id": "...", "market": "SE", "locale": "en_US"}

        Why market="SE"?
          Tink is Swedish. Swedish sandbox data is the richest.
          Norwegian real transactions also mix English (REMA 1000, Vipps, Netflix) —
          Swedish sandbox data is the same bilingual pattern.
        """
        response = httpx.post(
            f"{TINK_BASE_URL}/api/v1/user/create",
            headers=self._client_headers(),
            json={
                "external_user_id": external_user_id,
                "market": "SE",
                "locale": "en_US",
            },
            timeout=30.0,
        )
        # 409 = user already exists — safe to ignore, continue with existing user
        if response.status_code == 409:
            return
        response.raise_for_status()

    def get_user_token(self, external_user_id: str = SANDBOX_USER_EXTERNAL_ID) -> None:
        """
        Get an access token scoped to the sandbox user's accounts.

        Two-step process (why two steps? — security isolation):
          Step 1: Create authorization grant → get a short-lived one-use code
            POST /api/v1/oauth/authorization-grant
            Headers: Authorization: Bearer {client_token}
            Body: {"external_user_id": "...", "scope": "accounts:read,transactions:read"}
            Response: {"code": "abc123..."}

          Step 2: Exchange code for user access token
            POST /api/v1/oauth/token
            Body: grant_type=authorization_code, code={from step 1}, client_id, client_secret
            Response: {"access_token": "eyJ...", "expires_in": 1799}

        The client_token has admin scope (can create users).
        The user_token has limited scope (can only read this user's accounts).
        Even if a user_token leaks, attacker cannot manage other users.
        """
        # Step 1: Create authorization grant code
        grant_response = httpx.post(
            f"{TINK_BASE_URL}/api/v1/oauth/authorization-grant",
            headers=self._client_headers(),
            data={                                   # form-encoded, NOT json — Tink OAuth endpoints use form data
                "external_user_id": external_user_id,
                "scope": "accounts:read transactions:read credentials:read credentials:write",
                "actor_client_id": self._client_id,
            },
            timeout=30.0,
        )
        if not grant_response.is_success:
            raise RuntimeError(
                f"authorization-grant failed {grant_response.status_code}: {grant_response.text}"
            )
        code = grant_response.json()["code"]

        # Step 2: Exchange one-use code for user access token
        token_response = httpx.post(
            f"{TINK_BASE_URL}/api/v1/oauth/token",
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "authorization_code",
                "code": code,
            },
            timeout=30.0,
        )
        token_response.raise_for_status()
        self._user_token = token_response.json()["access_token"]

    # ── Sandbox Bank Connection ────────────────────────────────────────────

    def connect_sandbox_bank(self) -> str:
        """
        Link the Tink test provider to the sandbox user.

        In production: user would be redirected to their real bank login page.
        In sandbox: se-test-open-banking-redirect auto-completes with mock data.

        POST /api/v1/credentials
        Headers: Authorization: Bearer {user_token}
        Body: {"providerName": "se-test-open-banking-redirect", "fields": {}}
        Response: {"id": "cred_id...", "status": "CREATED", ...}

        Returns:
            credentials_id — used to poll sync status
        """
        response = httpx.post(
            f"{TINK_BASE_URL}/api/v1/credentials",
            headers=self._user_headers(),
            json={
                "providerName": "se-test-open-banking-redirect",
                "fields": {},
            },
            timeout=30.0,
        )
        # 409 = credentials for this provider already exist — already synced
        if response.status_code == 409:
            print("[tink] Bank already connected — fetching existing credentials ID")
            existing = httpx.get(
                f"{TINK_BASE_URL}/api/v1/credentials",
                headers=self._user_headers(),
                timeout=30.0,
            )
            existing.raise_for_status()
            creds = existing.json().get("credentials", [])
            if creds:
                return creds[0]["id"]
            raise RuntimeError("409 but no existing credentials found")

        if not response.is_success:
            raise RuntimeError(
                f"create credentials failed {response.status_code}: {response.text}"
            )
        return response.json()["id"]

    def wait_for_bank_sync(self, credentials_id: str, timeout_seconds: int = 60) -> None:
        """
        Poll credentials status until Tink finishes ingesting mock bank data.

        Status lifecycle:
          CREATED → UPDATING → UPDATED  (success — transactions now available)
                             → AUTHENTICATION_ERROR  (sandbox provider issue)

        Typical sandbox sync time: 5-15 seconds.

        GET /api/v1/credentials/{id}
        Headers: Authorization: Bearer {user_token}
        Response: {"id": "...", "status": "UPDATING", ...}
        """
        print("[tink] Waiting for sandbox bank sync...", end="", flush=True)
        for _ in range(timeout_seconds // 3):
            response = httpx.get(
                f"{TINK_BASE_URL}/api/v1/credentials/{credentials_id}",
                headers=self._user_headers(),
                timeout=10.0,
            )
            response.raise_for_status()
            status = response.json().get("status", "")
            print(f" {status}", end="", flush=True)

            if status == "UPDATED":
                print()  # newline after status dots
                return
            if status in ("AUTHENTICATION_ERROR", "DELETED", "SESSION_EXPIRED"):
                print()
                raise RuntimeError(f"Bank sync failed: {status}")

            time.sleep(3)

        print()
        raise RuntimeError(f"Bank sync timed out after {timeout_seconds}s")

    # ── Data Fetching ──────────────────────────────────────────────────────

    def get_accounts(self) -> list[dict]:
        """
        List all accounts for the sandbox user.

        GET /data/v2/accounts
        Headers: Authorization: Bearer {user_token}

        Response:
            {
                "accounts": [
                    {
                        "id": "a3f8c2...",
                        "name": "Checking Account",
                        "type": "CHECKING",
                        "balances": {
                            "available": {
                                "value": {"unscaledValue": "100000", "scale": "2"}
                            }
                        }
                    }
                ]
            }
        """
        response = httpx.get(
            f"{TINK_BASE_URL}/data/v2/accounts",
            headers=self._user_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("accounts", [])

    def get_transactions(self, page_size: int = 100) -> list[TinkTransaction]:
        """
        Fetch all transactions for the sandbox user across all accounts.

        Tink uses cursor-based pagination — each response includes a nextPageToken
        if there are more pages. We loop until nextPageToken is absent.

        GET /data/v2/transactions?pageSize=100
        GET /data/v2/transactions?pageSize=100&pageToken={nextPageToken}

        Args:
            page_size: Transactions per API call (max 100 per Tink docs)

        Returns:
            All available sandbox transactions
        """
        transactions = []
        page_token: Optional[str] = None

        while True:
            params: dict = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            response = httpx.get(
                f"{TINK_BASE_URL}/data/v2/transactions",
                headers=self._user_headers(),
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            for raw_tx in data.get("transactions", []):
                transactions.append(self._parse_transaction(raw_tx))

            page_token = data.get("nextPageToken")
            if not page_token:
                break  # Last page reached

        return transactions

    @staticmethod
    def _parse_transaction(raw: dict) -> TinkTransaction:
        amount_data = raw.get("amount", {}).get("value", {})
        unscaled = int(amount_data.get("unscaledValue", "0"))
        scale = int(amount_data.get("scale", "0"))
        amount = abs(Decimal(unscaled) / Decimal(10 ** scale))

        currency = raw.get("amount", {}).get("currencyCode", "SEK")

        descriptions = raw.get("descriptions", {})
        description = (
            descriptions.get("display")
            or descriptions.get("original")
            or "UNKNOWN"
        )

        date_str = raw.get("dates", {}).get("booked", "2024-01-01")
        booking_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

        return TinkTransaction(
            transaction_id=raw.get("id", str(uuid4())),
            description=description,
            amount=amount,
            currency=currency,
            booking_date=booking_date,
            raw=raw,
        )



def exchange_code_for_transactions(
    client_id: str,
    client_secret: str,
    authorization_code: str,
) -> list[dict]:
    """
    Exchange a Tink Link authorization code for transactions.

    After completing the Tink Link browser flow, the redirect URL contains:
      ?code=fcad441a...&credentialsId=0f2483ad...

    The code is ONE-TIME USE and expires in ~5 minutes.
    Call this function immediately after getting the code.

    Steps:
      1. POST /api/v1/oauth/token with grant_type=authorization_code
         → get user access token
      2. GET /data/v2/transactions with the user token
         → fetch all transactions from the demo bank account

    Args:
        authorization_code: The 'code' param from the redirect URL after Tink Link flow
    """
    # Step 1: Exchange the one-time code for a user access token
    token_response = httpx.post(
        f"{TINK_BASE_URL}/api/v1/oauth/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "authorization_code",
            "code": authorization_code,
        },
        timeout=30.0,
    )
    if not token_response.is_success:
        raise RuntimeError(
            f"code exchange failed {token_response.status_code}: {token_response.text}"
        )
    user_token = token_response.json()["access_token"]
    print("[tink] Code exchanged → user token obtained")

    # Step 2: Fetch all transactions with cursor-based pagination
    transactions = []
    page_token: Optional[str] = None

    while True:
        params: dict = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token

        response = httpx.get(
            f"{TINK_BASE_URL}/data/v2/transactions",
            headers={"Authorization": f"Bearer {user_token}"},
            params=params,
            timeout=30.0,
        )
        if not response.is_success:
            raise RuntimeError(
                f"fetch transactions failed {response.status_code}: {response.text}"
            )
        data = response.json()
        transactions.extend(data.get("transactions", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    print(f"[tink] Fetched {len(transactions)} transactions via authorization code")

    return [
        {
            "event_id": tx.get("id", str(uuid4())),
            "description": (
                tx.get("descriptions", {}).get("display")
                or tx.get("descriptions", {}).get("original")
                or "UNKNOWN"
            ),
            "amount": abs(
                Decimal(tx.get("amount", {}).get("value", {}).get("unscaledValue", "0"))
                / Decimal(10 ** int(tx.get("amount", {}).get("value", {}).get("scale", "0")))
            ),
            "currency": tx.get("amount", {}).get("currencyCode", "SEK"),
            "created_at": datetime.strptime(
                tx.get("dates", {}).get("booked", "2024-01-01"), "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc),
            "source": "tink_link",
        }
        for tx in transactions
    ]


def get_account_check_transactions(
    client_id: str,
    client_secret: str,
    report_id: str,
) -> list[dict]:
    """
    Fetch transactions from a completed Account Check + Transactions report.

    HOW THIS WORKS:
    ───────────────
    After a user completes the Tink Link browser flow (selecting their bank
    and logging in), Tink creates an Account Check report containing the
    user's account data + transaction history.

    The report ID arrives in the browser callback URL:
      https://console.tink.com/callback?account_verification_report_id=107291d3...

    This is the CORRECT modern Tink API — replaces the deprecated /api/v1/credentials flow.

    GET /api/v1/account-verification-reports/{reportId}/transactions
    Authorization: Bearer {client_token}  (scope: account-verification-reports:read)

    Args:
        report_id: From the callback URL after completing Tink Link flow.
                   Store in .env as TINK_REPORT_ID

    Returns:
        List of transaction dicts in the same format as collect_training_data()
    """
    client = TinkClient(client_id=client_id, client_secret=client_secret)
    client.authenticate_as_client()

    # Try the transactions sub-endpoint first
    response = httpx.get(
        f"{TINK_BASE_URL}/api/v1/account-verification-reports/{report_id}/transactions",
        headers=client._client_headers(),
        timeout=30.0,
    )

    if not response.is_success:
        # Fallback: fetch the full report (transactions may be embedded)
        report_response = httpx.get(
            f"{TINK_BASE_URL}/api/v1/account-verification-reports/{report_id}",
            headers=client._client_headers(),
            timeout=30.0,
        )
        if not report_response.is_success:
            raise RuntimeError(
                f"fetch report failed {report_response.status_code}: {report_response.text}"
            )
        data = report_response.json()
        raw_transactions = data.get("transactions", [])
        print(f"[tink] Report keys: {list(data.keys())}")  # debug: see what's in the report
    else:
        raw_transactions = response.json().get("transactions", [])

    print(f"[tink] Account Check report: {len(raw_transactions)} transactions")

    return [
        {
            "event_id": tx.get("id", str(uuid4())),
            "description": (
                tx.get("descriptions", {}).get("display")
                or tx.get("descriptions", {}).get("original")
                or tx.get("description", "UNKNOWN")
            ),
            "amount": abs(Decimal(str(
                int(tx.get("amount", {}).get("value", {}).get("unscaledValue", "0"))
                / (10 ** int(tx.get("amount", {}).get("value", {}).get("scale", "0")))
            ))),
            "currency": tx.get("amount", {}).get("currencyCode", "NOK"),
            "created_at": datetime.strptime(
                tx.get("dates", {}).get("booked", "2024-01-01"), "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc),
            "source": "tink_account_check",
        }
        for tx in raw_transactions
    ]


def collect_training_data(
    client_id: str,
    client_secret: str,
    external_user_id: str = SANDBOX_USER_EXTERNAL_ID,
) -> list[dict]:
    """
    High-level function: connect to Tink and return sandbox transactions as dicts.

    Same output format as psd2_collector.collect_training_data() so
    prepare_dataset.py can use either source interchangeably.

    Args:
        client_id:        Tink application client ID
        client_secret:    Tink application client secret
        external_user_id: Sandbox user identifier (fixed = reproducible data)

    Returns:
        List of transaction dicts:
            {
                "event_id": "tink-transaction-id",
                "description": "Spotify AB",
                "amount": Decimal("9.99"),
                "currency": "SEK",
                "created_at": datetime(..., tzinfo=UTC),
                "source": "tink_sandbox",
            }
    """
    client = TinkClient(client_id=client_id, client_secret=client_secret)

    # Step 1: Authenticate as your application (not as a user)
    client.authenticate_as_client()

    # Step 2: Create sandbox user (idempotent — safe to call repeatedly)
    client.create_sandbox_user(external_user_id)

    # Step 3: Get a token scoped to this user's accounts only
    client.get_user_token(external_user_id)

    # Step 4: Connect a test bank provider (seeds the sandbox user with mock transactions)
    # Idempotent — if already connected, fetches the existing credentials ID
    credentials_id = client.connect_sandbox_bank()
    client.wait_for_bank_sync(credentials_id)

    # Step 5: Fetch all sandbox transactions (now populated by the bank sync)
    transactions = client.get_transactions()

    return [
        {
            "event_id": tx.transaction_id,
            "description": tx.description,
            "amount": tx.amount,
            "currency": tx.currency,
            "created_at": tx.booking_date,
            "source": "tink_sandbox",
        }
        for tx in transactions
    ]
