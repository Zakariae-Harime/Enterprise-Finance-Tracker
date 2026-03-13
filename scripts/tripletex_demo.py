"""
Tripletex Sandbox Demo Script
==============================
Demonstrates the full Finance Tracker ↔ Tripletex integration.

Steps shown:
  1. Authenticate with Tripletex sandbox
  2. Push a consulting expense (Deloitte, 12,500 NOK)
  3. Push a SaaS expense (Salesforce, 4,800 NOK)
  4. Pull all vouchers back from the sandbox
  5. Print a formatted summary

Usage:
  set TRIPLETEX_CONSUMER_TOKEN=your-consumer-token
  set TRIPLETEX_EMPLOYEE_TOKEN=your-employee-token
  python scripts/tripletex_demo.py
"""
import asyncio
import os
import time
from decimal import Decimal
from uuid import uuid4

from src.accounting.tripletex import TripletexAdapter
from src.integrations.models import ERPExpense

CONSUMER_TOKEN = os.getenv("TRIPLETEX_CONSUMER_TOKEN")
EMPLOYEE_TOKEN = os.getenv("TRIPLETEX_EMPLOYEE_TOKEN")

DEMO_EXPENSES = [
    ERPExpense(
        expense_id=uuid4(),
        amount=Decimal("12500.00"),
        currency="NOK",
        description="Q1 Strategy Consulting — Project Apollo",
        merchant_name="Deloitte AS",
        expense_date="2026-03-13",
        category="consulting_services",
    ),
    ERPExpense(
        expense_id=uuid4(),
        amount=Decimal("4800.00"),
        currency="NOK",
        description="Salesforce CRM — Annual License",
        merchant_name="Salesforce EMEA",
        expense_date="2026-03-13",
        category="saas_software",
    ),
    ERPExpense(
        expense_id=uuid4(),
        amount=Decimal("28900.00"),
        currency="NOK",
        description="Azure Infrastructure — Q1 Cloud Costs",
        merchant_name="Microsoft Azure",
        expense_date="2026-03-13",
        category="cloud_infrastructure",
    ),
]


def _hr():
    print("─" * 60)


def _step(n: int, text: str):
    print(f"\n  Step {n}: {text}")


async def run_demo():
    if not CONSUMER_TOKEN or not EMPLOYEE_TOKEN:
        print("\n  ERROR: Set environment variables first:")
        print("    set TRIPLETEX_CONSUMER_TOKEN=your-token")
        print("    set TRIPLETEX_EMPLOYEE_TOKEN=your-token")
        return

    _hr()
    print("  Finance Tracker — Tripletex Sandbox Demo")
    print("  Target: api.tripletex.io (test environment)")
    _hr()

    adapter = TripletexAdapter(
        consumer_token=CONSUMER_TOKEN,
        employee_token=EMPLOYEE_TOKEN,
        use_sandbox=True,
    )

    # ── Step 1: Authenticate ───────────────────────────────────────────────
    _step(1, "Authenticating with Tripletex sandbox...")
    t0 = time.perf_counter()
    try:
        token = await adapter._get_session_token()
        ms = int((time.perf_counter() - t0) * 1000)
        print(f"     Session token obtained in {ms}ms")
        print(f"     Token: {token[:20]}...")
    except Exception as exc:
        print(f"     FAILED: {exc}")
        return

    # ── Step 2: Push expenses ──────────────────────────────────────────────
    _step(2, f"Pushing {len(DEMO_EXPENSES)} expenses to Tripletex...")
    pushed_ids = []
    total = Decimal("0")

    for expense in DEMO_EXPENSES:
        t0 = time.perf_counter()
        result = await adapter.push_expense(expense)
        ms = int((time.perf_counter() - t0) * 1000)

        if result.success:
            pushed_ids.append(result.external_id)
            total += expense.amount
            print(f"     ✓ {expense.merchant_name:<25} {expense.amount:>10} NOK  →  voucher #{result.external_id}  ({ms}ms)")
        else:
            print(f"     ✗ {expense.merchant_name:<25} FAILED: {result.error}")

    print(f"\n     Total pushed: {total:,} NOK across {len(pushed_ids)} vouchers")

    # ── Step 3: Pull back ──────────────────────────────────────────────────
    _step(3, "Pulling vouchers from Tripletex (since 2026-03-13)...")
    t0 = time.perf_counter()
    invoices = await adapter.pull_invoices("2026-03-13")
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"     Retrieved {len(invoices)} voucher(s) in {ms}ms")

    # ── Step 4: Verify roundtrip ───────────────────────────────────────────
    _step(4, "Verifying bidirectional sync...")
    pulled_ids = {inv.external_id for inv in invoices}
    verified = [eid for eid in pushed_ids if eid in pulled_ids]
    print(f"     Pushed: {pushed_ids}")
    print(f"     Found in pull: {verified}")

    if len(verified) == len(pushed_ids):
        print(f"\n     ALL {len(verified)} vouchers confirmed in Tripletex")
    else:
        missing = set(pushed_ids) - pulled_ids
        print(f"\n     WARNING: {len(missing)} voucher(s) not yet visible (replication lag?): {missing}")

    # ── Summary ────────────────────────────────────────────────────────────
    _hr()
    print("  DEMO COMPLETE")
    print(f"  Pushed:   {len(pushed_ids)} expenses  ({total:,} NOK)")
    print(f"  Pulled:   {len(invoices)} vouchers from Tripletex")
    print(f"  Verified: {len(verified)}/{len(pushed_ids)} roundtrip confirmed")
    print(f"  API:      api.tripletex.io (sandbox)")
    _hr()


if __name__ == "__main__":
    asyncio.run(run_demo())
