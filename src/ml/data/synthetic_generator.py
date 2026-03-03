"""
Synthetic Norwegian transaction generator.

Generates realistic training data when GoCardless sandbox isn't sufficient.
GoCardless sandbox → ~100 transactions (same each run, not enough).
Synthetic generator → 10,000+ varied transactions in under 1 second.

Why does variation matter?
  The model needs to generalize, not memorize.
  If it only sees "REMA 1000 OSLO", it might fail on "REMA 1000 BERGEN".
  By generating many location variants, the model learns that location
  suffixes don't change the category — the merchant name does.

Format mimics real bank transaction descriptions:
  "REMA 1000 MAJORSTUEN    250.00 NOK"
  "SAP AG SOFTWARE LICENSE NOK 15000.00"
  "ACCENTURE NORGE AS KONSULENTBISTAND 87500.00"
"""
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from src.ml.data.label_transactions import CATEGORIES


# ─── Oslo + Norwegian City Suffixes ───────────────────────────────────────────
# Real transactions often include store locations.
# Including these teaches the model that location != category signal.
OSLO_AREAS = [
    "MAJORSTUEN", "GRÜNERLØKKA", "FROGNER", "AKER BRYGGE",
    "TJUVHOLMEN", "SENTRUM", "TORSHOV", "BISLETT", "BOGSTADVEIEN",
    "BRYN", "HELSFYR", "LYSAKER", "SKØYEN", "NYDALEN", "CARL BERNER",
]

NORWEGIAN_CITIES = [
    "OSLO", "BERGEN", "TRONDHEIM", "STAVANGER", "TROMSØ",
    "KRISTIANSAND", "FREDRIKSTAD", "DRAMMEN", "SANDNES", "BODØ",
    "ÅLESUND", "HAMAR", "SARPSBORG", "SKIEN", "ASKER",
]

ALL_LOCATIONS = OSLO_AREAS + NORWEGIAN_CITIES


# ─── Merchant Templates per Category ──────────────────────────────────────────
# Each entry: (description_template, min_amount, max_amount)
# Amounts in NOK. Range is realistic for that merchant type.
#
# Example: REMA 1000 — typical grocery run is 80-1200 NOK.
# SAP AG license — typical monthly is 5,000-50,000 NOK.
# Accenture consulting — typical monthly invoice 50,000-500,000 NOK.

MERCHANT_TEMPLATES: dict[str, list[tuple[str, float, float]]] = {

    "groceries": [
        ("REMA 1000 {location}", 80.0, 1200.0),
        ("KIWI {location}", 60.0, 800.0),
        ("MENY {location}", 150.0, 2500.0),
        ("SPAR {location}", 50.0, 600.0),
        ("COOP PRIX {location}", 70.0, 900.0),
        ("COOP EXTRA {location}", 100.0, 1800.0),
        ("BUNNPRIS {location}", 40.0, 500.0),
        ("JOKER {location}", 30.0, 400.0),
        ("EUROSPAR {location}", 80.0, 700.0),
        ("NARVESEN {location}", 15.0, 150.0),
        ("DELI DE LUCA {location}", 25.0, 200.0),
        ("COOP OBS {location}", 200.0, 5000.0),        # Hypermarket — larger amounts
    ],

    "fuel": [
        ("CIRCLE K {location}", 400.0, 1500.0),
        ("SHELL {location}", 350.0, 1400.0),
        ("ESSO {location}", 380.0, 1350.0),
        ("ST1 {location}", 360.0, 1300.0),
        ("UNO-X {location}", 320.0, 1200.0),
        ("BEST {location}", 300.0, 1100.0),
        ("YX ENERGI {location}", 340.0, 1250.0),
        ("CIRCLE K DRIVSTOFF {location}", 450.0, 1600.0),
    ],

    "software": [
        ("SAP AG SOFTWARE LICENSE", 5000.0, 50000.0),
        ("SAP SE LIZENZ", 8000.0, 80000.0),
        ("MICROSOFT 365 BUSINESS", 200.0, 5000.0),
        ("MICROSOFT AZURE SERVICES", 1000.0, 30000.0),
        ("ORACLE DATABASE LICENSE", 10000.0, 100000.0),
        ("SALESFORCE CRM SUBSCRIPTION", 3000.0, 25000.0),
        ("AWS AMAZON WEB SERVICES", 500.0, 40000.0),
        ("ATLASSIAN JIRA CONFLUENCE", 200.0, 3000.0),
        ("GITHUB ENTERPRISE", 300.0, 5000.0),
        ("SLACK BUSINESS ABONNEMENT", 100.0, 2000.0),
        ("ADOBE CREATIVE CLOUD LISENS", 500.0, 8000.0),
        ("SERVICENOW IT PLATFORM", 5000.0, 60000.0),
        ("WORKDAY HCM SUBSCRIPTION", 8000.0, 80000.0),
        ("DATADOG MONITORING SaaS", 1000.0, 15000.0),
        ("SNOWFLAKE DATA CLOUD", 2000.0, 25000.0),
    ],

    "consulting": [
        ("ACCENTURE NORGE AS KONSULENTBISTAND", 50000.0, 500000.0),
        ("DELOITTE AS RÅDGIVNING", 30000.0, 300000.0),
        ("PWC ADVISORY SERVICES", 25000.0, 250000.0),
        ("KPMG CONSULTING", 20000.0, 200000.0),
        ("EY ERNST YOUNG ADVISORY", 25000.0, 220000.0),
        ("BOUVET AS IT-KONSULENT", 80000.0, 400000.0),
        ("CAPGEMINI NORGE KONSULENTBISTAND", 60000.0, 350000.0),
        ("KNOWIT SOLUTIONS OSLO", 70000.0, 300000.0),
        ("BEKK CONSULTING AS", 90000.0, 450000.0),
        ("SOPRA STERIA AS", 65000.0, 320000.0),
        ("TIETOEVRY IT TJENESTER", 40000.0, 250000.0),
        ("CGI NORGE AS KONSULENT", 55000.0, 280000.0),
        ("MCKINSEY COMPANY OSLO", 150000.0, 1000000.0),
        ("BCG BOSTON CONSULTING NORGE", 120000.0, 800000.0),
    ],

    "dining": [
        ("MCDONALDS {location}", 80.0, 600.0),
        ("BURGER KING {location}", 70.0, 500.0),
        ("SUBWAY {location}", 60.0, 300.0),
        ("STARBUCKS {location}", 40.0, 200.0),
        ("ESPRESSO HOUSE {location}", 35.0, 250.0),
        ("WAYNES COFFEE {location}", 30.0, 200.0),
        ("COFFEE HOUSE ONE {location}", 25.0, 150.0),
        ("RESTAURANT {location}", 150.0, 3000.0),
        ("SUSHI RESTAURANT {location}", 200.0, 1500.0),
        ("THAI RESTAURANT {location}", 120.0, 800.0),
        ("PIZZERIA {location}", 100.0, 600.0),
        ("CATERING SELSKAP AS", 2000.0, 25000.0),     # B2B catering
    ],

    "transport": [
        ("RUTER BILLETT OSLO", 38.0, 950.0),          # Single ticket or monthly pass
        ("VY NSB TOGBILLETT", 80.0, 1500.0),
        ("SAS AIRLINES BILLETT", 500.0, 8000.0),
        ("NORWEGIAN AIR BILLETT", 300.0, 5000.0),
        ("WIDEROE BILLETT", 400.0, 6000.0),
        ("FLYTOGET {location}", 95.0, 190.0),          # Airport express — fixed routes
        ("FLYBUSSEN {location}", 120.0, 250.0),
        ("UBER TECHNOLOGIES {location}", 80.0, 500.0),
        ("BOLT RIDE {location}", 60.0, 400.0),
        ("AUTOPASS BOMPENGER", 20.0, 200.0),          # Road tolls
        ("PARKERING {location}", 50.0, 400.0),
        ("ENTUR MOBILITET", 38.0, 950.0),
    ],

    "utilities": [
        ("TELENOR BEDRIFT MOBILABONNEMENT", 300.0, 2000.0),
        ("TELIA NORGE ABONNEMENT", 250.0, 1800.0),
        ("ICE.NET MOBILABONNEMENT", 200.0, 1000.0),
        ("HAFSLUND NETT NETTLEIE", 500.0, 5000.0),    # Electricity grid
        ("FJORDKRAFT STRØM", 800.0, 8000.0),           # Electricity consumption
        ("LYSE ENERGI STAVANGER", 600.0, 6000.0),
        ("TIBBER ELECTRICITY", 700.0, 7000.0),
        ("ALTIBOX BREDBÅND", 400.0, 1200.0),
        ("GET KABELTV INTERNETT", 350.0, 1100.0),
        ("NEXTGENTEL BREDBÅND AS", 300.0, 900.0),
        ("OSLO KOMMUNE RENOVASJON", 200.0, 800.0),    # Waste collection
    ],
}


def _random_amount(min_nok: float, max_nok: float) -> Decimal:
    """Generate a realistic transaction amount with 2 decimal places."""
    amount = random.uniform(min_nok, max_nok)
    return Decimal(str(round(amount, 2)))


def _random_date(days_back: int = 365) -> datetime:
    """Generate a random transaction date within the last N days."""
    offset = random.randint(0, days_back)
    return datetime.now(timezone.utc) - timedelta(days=offset)


def generate_transactions(
    n_per_category: int = 500,
    seed: int = 42,
) -> list[dict]:
    """
    Generate synthetic Norwegian transaction records for training.

    Each generated transaction has the same schema as a real transaction
    from GoCardless or our event store — the training pipeline doesn't
    know (or care) whether data is real or synthetic.

    Args:
        n_per_category: Number of transactions to generate per category.
                        Default 500 → 3,500 total (7 categories × 500).
                        Use 1000 for better model performance.
        seed: Random seed for reproducibility.
              Same seed → same dataset every run.
              Change seed to get a different dataset variant.

    Returns:
        List of transaction dicts, randomly shuffled.

    Example output record:
        {
            "event_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "description": "REMA 1000 MAJORSTUEN",
            "amount": Decimal("340.50"),
            "currency": "NOK",
            "category": "groceries",     ← the training label
            "created_at": datetime(...),
            "tenant_id": "00000000-...",
            "is_synthetic": True,        ← mark so we can filter in analysis
        }
    """
    random.seed(seed)

    transactions = []
    tenant_id = "00000000-0000-0000-0000-000000000001"

    for category, templates in MERCHANT_TEMPLATES.items():
        generated_for_category = 0
        template_idx = 0

        while generated_for_category < n_per_category:
            # Cycle through templates — distribute evenly across all merchants
            template, min_amt, max_amt = templates[template_idx % len(templates)]
            template_idx += 1

            # Fill in {location} placeholder if present
            if "{location}" in template:
                location = random.choice(ALL_LOCATIONS)
                description = template.format(location=location)
            else:
                description = template

            transaction = {
                "event_id": str(uuid4()),
                "description": description,
                "amount": _random_amount(min_amt, max_amt),
                "currency": "NOK",
                "category": category,          # ← ground truth label
                "created_at": _random_date(),
                "tenant_id": tenant_id,
                "is_synthetic": True,
            }
            transactions.append(transaction)
            generated_for_category += 1

    # Shuffle so categories aren't grouped — important for training
    # Without shuffling, the model sees all "groceries" then all "fuel" etc.
    # This creates biased gradient updates and slower convergence.
    random.shuffle(transactions)

    return transactions


def get_dataset_stats(transactions: list[dict]) -> dict:
    """
    Print distribution statistics for a generated dataset.

    Always check this before training — if one category has 10x more examples
    than another (class imbalance), you need to handle it:
      Option 1: Oversample minority class
      Option 2: Use class weights in loss function (what we do)
      Option 3: Undersample majority class

    Example output:
        {
            "total": 3500,
            "by_category": {
                "groceries": 500,
                "fuel": 500,
                ...
            },
            "is_balanced": True   ← True if max/min ratio < 2.0
        }
    """
    by_category: dict[str, int] = {}
    for tx in transactions:
        cat = tx["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    counts = list(by_category.values())
    is_balanced = max(counts) / min(counts) < 2.0 if counts else True

    return {
        "total": len(transactions),
        "by_category": by_category,
        "is_balanced": is_balanced,
    }
