"""
Synthetic enterprise B2B transaction generator.

Generates realistic training data for the transaction categorizer.
Mimics real Norwegian enterprise bank statement descriptions:
  "AMAZON WEB SERVICES EMEA SARL NOK 45231.50"
  "DELOITTE AS RÅDGIVNING KONSULENTBISTAND"
  "SAS GROUP BILLETT OSLO-LONDON"
  "TELENOR BEDRIFT MOBILABONNEMENT FAKTURA"

Enterprise transactions differ from personal:
  - Amounts are larger (500-2,000,000 NOK range depending on category)
  - No location suffixes for SaaS/cloud (invoiced by HQ, not local store)
  - Descriptions include company legal names (AS, SARL, GmbH, Ltd)
  - Many vendors are international (converted to NOK by bank)
  - Recurring monthly invoices dominate

Why does variation matter for training?
  If the model only sees "AMAZON WEB SERVICES EMEA SARL", it may fail on
  "AWS EMEA CLOUD SERVICES". By generating multiple description variants
  per vendor, the model learns that the vendor name is the signal — not
  the exact surrounding words.
"""
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from src.ml.data.label_transactions import CATEGORIES


# ─── Travel Location Suffixes ─────────────────────────────────────────────────
# Only travel_expenses use {location} — flights/hotels have destination context.
# Other enterprise categories (SaaS, cloud) are invoiced by HQ — no location.

NORWEGIAN_AIRPORTS = [
    "OSLO GARDERMOEN", "BERGEN FLESLAND", "STAVANGER SOLA",
    "TRONDHEIM VAERNES", "TROMSO LANGNES",
]

EUROPEAN_CITIES = [
    "LONDON HEATHROW", "AMSTERDAM SCHIPHOL", "FRANKFURT",
    "PARIS CDG", "STOCKHOLM ARLANDA", "COPENHAGEN KASTRUP",
    "BRUSSELS", "ZURICH", "MUNICH", "DUBLIN",
]

NORWEGIAN_HOTEL_CITIES = [
    "OSLO", "BERGEN", "STAVANGER", "TRONDHEIM", "TROMSØ",
    "KRISTIANSAND", "ÅLESUND", "BODØ",
]

ALL_TRAVEL_LOCATIONS = NORWEGIAN_AIRPORTS + EUROPEAN_CITIES + NORWEGIAN_HOTEL_CITIES


# ─── Merchant Templates per Enterprise Category ───────────────────────────────
# Each entry: (description_template, min_amount_nok, max_amount_nok)
#
# Amount ranges reflect realistic enterprise invoice sizes:
#   - SaaS: 500-200,000 NOK (5-person startup vs. 500-person company, same product)
#   - Cloud: 5,000-500,000 NOK (usage-based, scales with engineering org)
#   - Consulting: 50,000-2,000,000 NOK (McKinsey strategy vs. junior IT contractor)
#   - Travel: 400-40,000 NOK (budget flight vs. business class + hotel)
#   - Marketing: 3,000-500,000 NOK (test campaign vs. national advertising)

MERCHANT_TEMPLATES: dict[str, list[tuple[str, float, float]]] = {

    "saas_software": [
        ("SALESFORCE.COM EMEA LTD SUBSCRIPTION", 5000.0, 200000.0),
        ("SALESFORCE CRM ENTERPRISE LICENSE", 8000.0, 150000.0),
        ("SLACK TECHNOLOGIES IRELAND LTD", 1000.0, 30000.0),
        ("SLACK BUSINESS PLUS SUBSCRIPTION", 800.0, 25000.0),
        ("GITHUB ENTERPRISE SERVER LICENSE", 3000.0, 80000.0),
        ("GITHUB TEAM SUBSCRIPTION", 500.0, 15000.0),
        ("ATLASSIAN JIRA SOFTWARE CLOUD", 2000.0, 50000.0),
        ("ATLASSIAN CONFLUENCE CLOUD", 1500.0, 40000.0),
        ("ATLASSIAN BITBUCKET PIPELINES", 1000.0, 20000.0),
        ("FIGMA ORGANIZATION PLAN", 2000.0, 30000.0),
        ("NOTION TEAM WORKSPACE", 500.0, 10000.0),
        ("MIRO COMPANY PLAN", 1000.0, 20000.0),
        ("ADOBE CREATIVE CLOUD ENTERPRISE LISENS", 3000.0, 60000.0),
        ("WORKDAY HCM ANNUAL SUBSCRIPTION", 50000.0, 500000.0),
        ("SERVICENOW IT SERVICE MGMT LICENSE", 20000.0, 300000.0),
        ("ZENDESK SUPPORT ENTERPRISE", 5000.0, 80000.0),
        ("DATADOG INFRASTRUCTURE MONITORING", 5000.0, 100000.0),
        ("SNOWFLAKE DATA CLOUD USAGE", 10000.0, 200000.0),
        ("SAP SE ENTERPRISE LISENS", 20000.0, 500000.0),
        ("SAP AG SOFTWARE LISENS FAKTURA", 15000.0, 400000.0),
        ("ORACLE CLOUD APPLICATIONS LICENSE", 25000.0, 300000.0),
        ("MICROSOFT 365 BUSINESS PREMIUM", 1000.0, 80000.0),
        ("AUTODESK AEC COLLECTION LISENS", 8000.0, 120000.0),
        ("MONDAY.COM ENTERPRISE PLAN", 2000.0, 40000.0),
        ("ASANA BUSINESS SUBSCRIPTION", 1500.0, 35000.0),
    ],

    "cloud_infrastructure": [
        ("AMAZON WEB SERVICES EMEA SARL", 5000.0, 500000.0),
        ("AWS EMEA CLOUD COMPUTE SERVICES", 8000.0, 400000.0),
        ("GOOGLE CLOUD PLATFORM EMEA", 5000.0, 300000.0),
        ("GOOGLE CLOUD COMPUTE ENGINE USAGE", 3000.0, 200000.0),
        ("MICROSOFT AZURE SERVICES IRELAND", 8000.0, 450000.0),
        ("AZURE SERVICES COMPUTE STORAGE", 5000.0, 350000.0),
        ("CLOUDFLARE INC BUSINESS PLAN", 1000.0, 30000.0),
        ("CLOUDFLARE TEAMS ZERO TRUST", 2000.0, 50000.0),
        ("DIGITALOCEAN CLOUD HOSTING", 500.0, 20000.0),
        ("HEROKU ENTERPRISE DYNOS", 2000.0, 50000.0),
        ("VERCEL PRO TEAM HOSTING", 500.0, 10000.0),
        ("FASTLY CDN SERVICES", 2000.0, 80000.0),
        ("AKAMAI CONTENT DELIVERY NETWORK", 5000.0, 150000.0),
        ("HETZNER CLOUD SERVER SERVICES", 300.0, 10000.0),
    ],

    "consulting_services": [
        ("ACCENTURE NORGE AS KONSULENTBISTAND", 100000.0, 1000000.0),
        ("DELOITTE AS RÅDGIVNING FAKTURA", 80000.0, 800000.0),
        ("KPMG AS ADVISORY SERVICES", 50000.0, 600000.0),
        ("PWC ADVISORY SERVICES OSLO", 70000.0, 700000.0),
        ("EY NORGE AS MANAGEMENT CONSULTING", 60000.0, 650000.0),
        ("MCKINSEY COMPANY OSLO STRATEGY", 200000.0, 2000000.0),
        ("BCG BOSTON CONSULTING GROUP NORGE", 180000.0, 1800000.0),
        ("BOUVET AS IT-KONSULENT BISTAND", 80000.0, 600000.0),
        ("CAPGEMINI NORGE KONSULENTBISTAND", 90000.0, 700000.0),
        ("KNOWIT SOLUTIONS AS OSLO", 70000.0, 500000.0),
        ("BEKK CONSULTING AS RÅDGIVNING", 90000.0, 800000.0),
        ("SOPRA STERIA AS IT-TJENESTER", 75000.0, 550000.0),
        ("TIETOEVRY NORGE KONSULENTBISTAND", 60000.0, 500000.0),
        ("CGI NORGE AS IT-KONSULENT", 65000.0, 480000.0),
        ("AVANADE NORWAY MICROSOFT CONSULTING", 80000.0, 600000.0),
        ("COMPUTAS AS DATA RÅDGIVNING", 70000.0, 450000.0),
    ],

    "travel_expenses": [
        ("SAS GROUP BILLETT {location}", 800.0, 15000.0),
        ("SAS AIRLINES BUSINESS CLASS {location}", 3000.0, 40000.0),
        ("NORWEGIAN AIR BILLETT {location}", 400.0, 8000.0),
        ("WIDERØE REGIONAL BILLETT {location}", 600.0, 5000.0),
        ("LUFTHANSA FLYBILLETT {location}", 1500.0, 20000.0),
        ("KLM ROYAL DUTCH AIRLINES {location}", 1200.0, 18000.0),
        ("BRITISH AIRWAYS BILLETT {location}", 1400.0, 22000.0),
        ("MARRIOTT HOTELS RESORTS {location}", 1500.0, 6000.0),
        ("HILTON HOTELS RESORTS {location}", 1200.0, 5500.0),
        ("RADISSON BLU HOTEL {location}", 1000.0, 4500.0),
        ("SCANDIC HOTELS {location}", 800.0, 3500.0),
        ("THON HOTELS {location}", 700.0, 3000.0),
        ("CLARION HOTEL {location}", 900.0, 4000.0),
        ("BOOKING.COM HOTELLBOOKING", 800.0, 6000.0),
        ("UBER BUSINESS TECHNOLOGIES", 100.0, 1000.0),
        ("FLYTOGET AIRPORT EXPRESS OSLO", 95.0, 190.0),
        ("FLYBUSSEN AIRPORT SHUTTLE {location}", 100.0, 250.0),
        ("BCD TRAVEL CORPORATE BOOKING", 500.0, 30000.0),
        ("CWT CARLSON WAGONLIT TRAVEL", 500.0, 25000.0),
        ("VY BILLETT BUSINESS TOGTUR", 150.0, 2000.0),
    ],

    "office_facilities": [
        ("STAPLES NORWAY AS KONTORREKVISITA", 500.0, 15000.0),
        ("OFFICELINE KONTORREKVISITA AS", 300.0, 10000.0),
        ("ELKJOP BEDRIFT KONTORUTSTYR", 1000.0, 50000.0),
        ("REGUS KONTORLEIE COWORKING", 5000.0, 30000.0),
        ("IWG SPACES WORKSPACE OSLO", 4000.0, 25000.0),
        ("WEWORK KONTORLEIE", 8000.0, 40000.0),
        ("KONICA MINOLTA NORGE KOPIMASKIN", 2000.0, 20000.0),
        ("RICOH NORGE PRINTER SERVICE", 1500.0, 15000.0),
        ("ISS FACILITY SERVICES NORGE", 5000.0, 50000.0),
        ("SODEXO CATERING OG RENHOLD", 3000.0, 30000.0),
        ("SECURITAS VAKTTJENESTER NORGE", 2000.0, 25000.0),
        ("POSTEN NORGE FORSENDELSE", 200.0, 5000.0),
        ("DHL EXPRESS NORGE PAKKE", 300.0, 8000.0),
    ],

    "marketing_advertising": [
        ("GOOGLE ADS IRELAND LIMITED", 5000.0, 500000.0),
        ("GOOGLE ANALYTICS PREMIUM", 2000.0, 50000.0),
        ("META PLATFORMS IRELAND ADS", 5000.0, 300000.0),
        ("META ADS FACEBOOK INSTAGRAM", 3000.0, 200000.0),
        ("LINKEDIN MARKETING SOLUTIONS", 5000.0, 150000.0),
        ("LINKEDIN ADS CAMPAIGN MANAGER", 3000.0, 100000.0),
        ("MAILCHIMP EMAIL MARKETING", 500.0, 20000.0),
        ("HUBSPOT MARKETING HUB", 5000.0, 100000.0),
        ("MARKETO ADOBE MARKETING AUTO", 10000.0, 200000.0),
        ("HOOTSUITE SOCIAL MEDIA MGMT", 1000.0, 20000.0),
        ("SEMRUSH SEO TOOLKIT", 1500.0, 15000.0),
        ("AHREFS SEO TOOLS", 1000.0, 12000.0),
        ("HOTJAR ANALYTICS", 500.0, 8000.0),
        ("ADFORM ADVERTISING PLATFORM", 5000.0, 200000.0),
        ("KLAVIYO EMAIL AUTOMATION", 800.0, 30000.0),
    ],

    "telecommunications": [
        ("TELENOR BEDRIFT MOBILABONNEMENT FAKTURA", 1000.0, 50000.0),
        ("TELENOR BEDRIFT BREDBÅND BEDRIFT", 500.0, 10000.0),
        ("TELIA BEDRIFT MOBILABONNEMENT", 800.0, 40000.0),
        ("TELIA COMPANY ENTERPRISE SERVICES", 1500.0, 60000.0),
        ("ICE KOMMUNIKASJON BEDRIFT 5G", 500.0, 20000.0),
        ("CISCO SYSTEMS NORWAY NETWORK", 5000.0, 200000.0),
        ("CISCO WEBEX TEAMS LICENSE", 2000.0, 80000.0),
        ("ZOOM VIDEO COMMUNICATIONS INC", 1000.0, 50000.0),
        ("ZOOM ENTERPRISE LICENSE", 3000.0, 80000.0),
        ("RINGCENTRAL CLOUD PBX", 2000.0, 40000.0),
        ("TELIO VOIP TJENESTER", 500.0, 10000.0),
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
    Generate synthetic enterprise B2B transaction records for training.

    Each generated transaction has the same schema as a real transaction
    from Tink or our event store — the training pipeline doesn't
    know (or care) whether data is real or synthetic.

    Args:
        n_per_category: Number of transactions to generate per category.
                        Default 500 → 3,500 total (7 categories × 500).
                        Use 1000 for better model generalization.
        seed: Random seed for reproducibility.
              Same seed → same dataset every run.

    Returns:
        List of transaction dicts, randomly shuffled.

    Example output record:
        {
            "event_id": "f47ac10b-...",
            "description": "AMAZON WEB SERVICES EMEA SARL",
            "amount": Decimal("45231.50"),
            "currency": "NOK",
            "category": "cloud_infrastructure",   ← ground truth label
            "created_at": datetime(...),
            "tenant_id": "00000000-...",
            "is_synthetic": True,
        }
    """
    random.seed(seed)

    transactions = []
    tenant_id = "00000000-0000-0000-0000-000000000001"

    for category, templates in MERCHANT_TEMPLATES.items():
        generated_for_category = 0
        template_idx = 0

        while generated_for_category < n_per_category:
            # Cycle through templates — distributes evenly across all vendors
            template, min_amt, max_amt = templates[template_idx % len(templates)]
            template_idx += 1

            # Only travel_expenses use {location} placeholders.
            # Non-travel templates get a unique invoice number suffix so that
            # drop_duplicates() in prepare_dataset.py does NOT collapse 500
            # identical copies of "SALESFORCE.COM EMEA LTD SUBSCRIPTION" down to 1.
            # The suffix is invisible to rule matching (keyword is still in the string).
            if "{location}" in template:
                location = random.choice(ALL_TRAVEL_LOCATIONS)
                description = template.format(location=location)
            else:
                invoice_num = random.randint(100000, 999999)
                description = f"{template} INV-{invoice_num}"

            transaction = {
                "event_id": str(uuid4()),
                "description": description,
                "amount": _random_amount(min_amt, max_amt),
                "currency": "NOK",
                "category": category,           # ← ground truth label
                "created_at": _random_date(),
                "tenant_id": tenant_id,
                "is_synthetic": True,
            }
            transactions.append(transaction)
            generated_for_category += 1

    # Shuffle so categories aren't grouped — prevents biased gradient updates
    random.shuffle(transactions)

    return transactions


def get_dataset_stats(transactions: list[dict]) -> dict:
    """
    Print distribution statistics for a generated dataset.

    Always check this before training — class imbalance degrades model quality.

    Example output:
        {
            "total": 3500,
            "by_category": {
                "saas_software": 500,
                "cloud_infrastructure": 500,
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
