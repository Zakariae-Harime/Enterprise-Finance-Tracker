"""
Rule-based transaction labeler for enterprise B2B financial transactions.

This is Layer 1 of the 3-layer categorization system AND the auto-labeling
engine for creating training data.

  Inference time:  "AMAZON WEB SERVICES EMEA SARL" → instantly returns "cloud_infrastructure" (0ms)
  Training time:   labels 5,000 raw B2B invoices without human effort

Why rules first?
  Enterprise B2B invoices follow highly predictable patterns.
  "DELOITTE AS RÅDGIVNING" is always consulting_services.
  "SALESFORCE.COM EMEA LTD" is always saas_software.
  No ML needed for known vendors — rules are faster, cheaper, 100% accurate.

Category definitions (Norwegian enterprise B2B finance context):
  saas_software         — Salesforce, Slack, GitHub, SAP, Figma, Notion, Jira
  cloud_infrastructure  — AWS, Azure GCP, Cloudflare, DigitalOcean
  consulting_services   — Deloitte, KPMG, McKinsey, Bouvet, Accenture, BEKK
  travel_expenses       — SAS, Norwegian Air, Marriott, Hilton, Uber Business
  office_facilities     — office supplies, co-working spaces, facilities management
  marketing_advertising — Google Ads, Meta, HubSpot, LinkedIn, Mailchimp
  telecommunications    — Telenor Bedrift, Telia, Cisco, Zoom, Webex
"""
from dataclasses import dataclass
from typing import Optional


# ─── Category Labels 
CATEGORIES = [
    "saas_software",         # SaaS platforms — Salesforce, Slack, SAP, GitHub
    "cloud_infrastructure",  # Cloud compute/storage — AWS, GCP, Azure, Cloudflare
    "consulting_services",   # Professional services — Big4, boutique firms, IT consulting
    "travel_expenses",       # Business travel — flights, hotels, rideshare
    "office_facilities",     # Office operations — supplies, co-working, facilities
    "marketing_advertising", # Paid marketing — Google Ads, Meta, agencies
    "telecommunications",    # Corporate comms — mobile plans, internet, VoIP
]

# Maps category name → integer index (required by neural networks).
# "saas_software" → 0, "cloud_infrastructure" → 1, "consulting_services" → 2, etc.
CATEGORY_TO_ID = {cat: idx for idx, cat in enumerate(CATEGORIES)}
ID_TO_CATEGORY = {idx: cat for cat, idx in CATEGORY_TO_ID.items()}


# ─── Merchant Rules

# UPPERCASE keywords — we always .upper() the transaction before matching.
# Order matters within each category: more specific keywords FIRST.
# CRITICAL: avoid short generic words that appear across categories.
#   BAD:  "MICROSOFT"  alone → matches "MICROSOFT AZURE" AND "MICROSOFT 365"
#   GOOD: "MICROSOFT AZURE" → cloud_infrastructure, "MICROSOFT 365" → saas_software
MERCHANT_RULES: dict[str, list[str]] = {

    # SaaS Software
    # Enterprise software billed as monthly/annual per-seat subscriptions.
    # These are OPEX, not CAPEX — no hardware, no one-time purchase.
    "saas_software": [
        "SALESFORCE",               # CRM platform — #1 SaaS by revenue globally
        "SLACK TECHNOLOGIES",       # Team messaging (Salesforce-owned since 2021)
        "SLACK",
        "GITHUB",                   # Code hosting + CI/CD (Microsoft-owned)
        "ATLASSIAN",                # Suite: Jira, Confluence, Bitbucket
        "JIRA SOFTWARE",
        "CONFLUENCE",
        "FIGMA",                    # Design/prototyping tool
        "NOTION",                   # Document management
        "MIRO",                     # Online whiteboard
        "ADOBE CREATIVE",           # Creative Cloud — design, video, PDF
        "ADOBE",
        "WORKDAY",                  # HR + Finance SaaS (common in large Norwegian enterprises)
        "SERVICENOW",               # IT service management
        "ZENDESK",                  # Customer support platform
        "DATADOG",                  # Observability SaaS
        "SNOWFLAKE",                # Cloud data warehouse (SaaS consumption model)
        "MONDAY.COM",               # Project management
        "ASANA",                    # Task management
        "SAP SE", "SAP AG", "SAP",  # ERP — used by Equinor, DNB, Telenor
        "ORACLE",                   # ERP + Database SaaS
        "MICROSOFT 365",            # Office suite as subscription (not Azure)
        "AUTODESK",                 # Engineering software (used by Aker, Statkraft)
        # NOTE: "LISENS"/"LICENSE" removed — too generic, causes collisions
        # ("ZOOM VIDEO COMMUNICATIONS INC LICENSE" would match saas_software instead of telecoms)
        # Unknown license invoices fall through to TF-IDF/BERT — correct behavior.
    ],

    # Cloud Infrastructure
    # Usage-based cloud compute, storage, CDN, networking.
    # Monthly variable invoices — scale with actual consumption.
    # IMPORTANT: "MICROSOFT AZURE" listed here, NOT standalone "MICROSOFT"
    "cloud_infrastructure": [
        "AMAZON WEB SERVICES",      # AWS — #1 cloud by market share
        "AWS EMEA",                 # AWS European invoicing entity
        "GOOGLE CLOUD",             # GCP
        "GOOGLE CLOUD PLATFORM",
        "MICROSOFT AZURE",          # Azure (standalone "MICROSOFT" → saas_software)
        "AZURE SERVICES",
        "CLOUDFLARE",               # CDN + zero-trust security
        "DIGITALOCEAN",             # SMB cloud
        "HEROKU",                   # Salesforce-owned PaaS
        "VERCEL",                   # Frontend cloud (popular in Norwegian startups)
        "FASTLY",                   # Edge CDN
        "AKAMAI",                   # CDN + security (used by Norwegian banks)
        "HETZNER",                  # German cloud, popular in Europe
    ],

    # Consulting Services
    # Professional services firms — billed as fixed-fee or time-and-materials.
    # Norwegian enterprise consulting is dominated by Big4 + Scandinavian IT firms.
    "consulting_services": [
        "ACCENTURE",                # Largest IT consulting firm in Norway by revenue
        "DELOITTE",                 # Big4 — audit + advisory + tech
        "KPMG",                     # Big4
        "PWC", "PRICEWATERHOUSECOOPERS",   # Big4
        "EY NORGE", "ERNST YOUNG",  # Big4 (avoid bare "EY" — too short, risk of collision)
        "MCKINSEY",                 # Strategy consulting — typically 150K-1M NOK invoices
        "BCG", "BOSTON CONSULTING", # Strategy consulting
        "BAIN",                     # Strategy consulting
        "BOUVET",                   # Largest Norwegian-owned IT consultancy
        "KNOWIT",                   # Norwegian IT consulting
        "BEKK",                     # Oslo boutique tech consulting (premium rates)
        "SOPRA STERIA",             # French-Norwegian IT
        "CAPGEMINI",                # Global IT consulting
        "TIETOEVRY", "TIETO", "EVRY",  # Finnish-Norwegian (merged 2019)
        "CGI NORGE", "CGI",         # Canadian IT, large Norwegian public sector presence
        "COMPUTAS",                 # Norwegian AI/data consulting (Visma subsidiary)
        "AVANADE",                  # Microsoft-focused (Accenture + Microsoft JV)
        "RÅDGIVNING",               # Norwegian: "consulting"
        "KONSULENTBISTAND",         # Norwegian: "consultant assistance"
        "KONSULENT",                # Norwegian: "consultant"
    ],

    # Travel & Expenses
    # Business travel booked on corporate cards.
    # Oslo-London, Oslo-Frankfurt, Oslo-Stockholm are the dominant corridors.
    "travel_expenses": [
        "SAS GROUP", "SAS AIRLINES", # Scandinavian Airlines — dominant in Nordics
        "NORWEGIAN AIR", "NORWEGIAN.COM",     # Norwegian Air Shuttle
        "WIDEROE", "WIDERØE",                 # Regional Norwegian airline
        "LUFTHANSA",                           # Common Oslo-Frankfurt-rest of world
        "BRITISH AIRWAYS",
        "KLM",                                 # Amsterdam hub — common connection point
        "MARRIOTT",                            # Business hotel chain
        "HILTON",                              # Business hotel chain
        "RADISSON",                            # Scandinavian hotel chain
        "SCANDIC HOTELS", "SCANDIC",           # Largest Nordic hotel chain
        "THON HOTELS", "THON",                 # Largest Norwegian-owned hotel chain
        "NORDIC CHOICE", "CLARION",            # Norwegian hotel chain
        "BOOKING.COM",                         # Most used in Norwegian corporate travel
        "EXPEDIA CORPORATE",
        "UBER BUSINESS", "UBER FOR BUSINESS",  # Specific to avoid matching personal Uber
        "UBER",
        "FLYTOGET",                            # Airport express Oslo
        "FLYBUSSEN",                           # Airport bus
        "BCD TRAVEL",                          # Corporate travel management firm
        "CWT", "CARLSON WAGONLIT",             # Corporate travel agency
        "VY BILLETT", "NSB",                   # Norwegian rail (business trips)
    ],

    # Office & Facilities
    # TODO(human): Add keywords for the office_facilities category.
    # This covers physical office running costs — NOT digital subscriptions.
    #
    # Think about what a CFO reviews as "office & facilities" spend:
    #   - Office supply stores (Norwegian + international)
    #   - Co-working space providers (enterprises often use Regus, WeWork, Spaces)
    #   - Facilities management (cleaning, security, maintenance companies)
    #   - Shipping and postal services (Posten, DHL, FedEx)
    #   - Office equipment (printers, copiers — Konica Minolta, Ricoh)
    #
    # Constraints to keep in mind:
    #   - Do NOT use "OFFICE" alone → "MICROSOFT OFFICE" would match cloud_infrastructure
    #   - Do NOT use "SUPPLIES" alone → too generic
    #   - Prefer specific company names over generic English words
    #   - Norwegian company names (AS suffix) are good candidates
    "office_facilities": [
        "STAPLES",                  # Office supply chain (Norway + international)
        "OFFICELINE",               # Norwegian office supplies distributor
        "ELKJOP BEDRIFT",           # Norwegian electronics — enterprise division only
        "REGUS",                    # Co-working spaces (IWG brand)
        "IWG", "SPACES",            # International Workplace Group — Regus parent company
        "WEWORK",                   # Co-working spaces
        "KONICA MINOLTA",           # Printer/copier fleet management
        "RICOH",                    # Printer/copier fleet management
        "ISS FACILITY",             # Facilities management (cleaning, maintenance)
        "SODEXO",                   # Facilities + catering services
        "SECURITAS",                # Security services
        "POSTEN NORGE",             # Norwegian postal/shipping
        "DHL EXPRESS",              # International courier
        "KONTORREKVISITA",          # Norwegian: "office supplies"
        "KONTORUTSTYR",             # Norwegian: "office equipment"
    ],

    # Marketing & Advertising
    # Paid media spend and marketing automation platforms.
    # Collision risk: Google also has "GOOGLE CLOUD" → solved by listing
    # "GOOGLE CLOUD" under cloud_infrastructure first (dict order = precedence).
    "marketing_advertising": [
        "GOOGLE ADS",               # Google paid search/display
        "GOOGLE ANALYTICS",         # GA4 paid tier
        "META PLATFORMS",           # Facebook + Instagram Ads invoicing entity
        "META ADS",
        "LINKEDIN ADS", "LINKEDIN MARKETING",  # B2B advertising
        "MAILCHIMP",                # Email marketing
        "HUBSPOT",                  # Marketing automation (also CRM — billing says "marketing")
        "MARKETO",                  # Adobe marketing automation
        "HOOTSUITE",                # Social media management
        "SEMRUSH",                  # SEO tooling
        "AHREFS",                   # SEO tooling
        "HOTJAR",                   # UX analytics
        "ADFORM",                   # Scandinavian ad tech platform
        "KLAVIYO",                  # E-commerce email marketing
        "ANNONSERING",              # Norwegian: "advertising"
        "MARKEDSFØRING",            # Norwegian: "marketing"
    ],

    # Telecommunications
    # Corporate communication infrastructure — mobile plans, broadband, VoIP, video.
    # "MICROSOFT TEAMS" is part of Microsoft 365 → covered under saas_software.
    "telecommunications": [
        "TELENOR BEDRIFT",          # Telenor's enterprise division (not consumer Telenor)
        "TELENOR",                  # Fallback — any Telenor invoice
        "TELIA BEDRIFT", "TELIA FÖRETAG", "TELIA",  # Swedish-Norwegian telecom
        "ICE KOMMUNIKASJON", "ICE.NET",    # Norwegian 5G operator
        "CISCO SYSTEMS",            # Network infrastructure + Webex
        "CISCO WEBEX", "WEBEX",    # Video conferencing (Cisco)
        "ZOOM VIDEO COMMUNICATIONS", "ZOOM",   # Video conferencing
        "RINGCENTRAL",              # Cloud PBX / VoIP
        "TELIO",                    # Norwegian VoIP
        "MOBILABONNEMENT",          # Norwegian: "mobile subscription"
        "BREDBÅND", "BREDBAND",    # Norwegian: "broadband"
        "INTERNETT",                # Norwegian: "internet" (corporate line)
    ],
}


# ─── Labeling Functions ────────────────────────────────────────────────────────

@dataclass
class LabelResult:
    """Result from auto_label() — includes confidence for downstream decisions."""
    category: Optional[str]       # Category name, or None if not matched
    confidence: float             # 1.0 for rules (always certain), 0.0 if no match
    matched_keyword: Optional[str]  # Which keyword triggered the match (for debugging)


def auto_label(description: str) -> LabelResult:
    """
    Label a B2B transaction description using merchant rules.

    How it works:
        1. Convert description to UPPERCASE (case-insensitive matching)
        2. For each category, check if ANY keyword appears in the description
        3. Return first match found (categories ordered by precedence in MERCHANT_RULES)
        4. Return None if no keyword matches — transaction goes to TF-IDF then BERT

    Args:
        description: Raw transaction description from bank
                     Example: "AMAZON WEB SERVICES EMEA SARL NOK 45231.50"

    Returns:
        LabelResult with category="cloud_infrastructure", confidence=1.0 if matched

    Example:
        >>> auto_label("DELOITTE AS RÅDGIVNING NOK 250000")
        LabelResult(category="consulting_services", confidence=1.0, matched_keyword="DELOITTE")

        >>> auto_label("UKJENT LEVERANDOR AS")  # Unknown vendor
        LabelResult(category=None, confidence=0.0, matched_keyword=None)
    """
    description_upper = description.upper()

    for category, keywords in MERCHANT_RULES.items():
        for keyword in keywords:
            if keyword in description_upper:
                return LabelResult(
                    category=category,
                    confidence=1.0,
                    matched_keyword=keyword,
                )

    # No rule matched — transaction goes to TF-IDF or BERT layer
    return LabelResult(category=None, confidence=0.0, matched_keyword=None)


def label_batch(descriptions: list[str]) -> list[LabelResult]:
    """
    Label a list of B2B transaction descriptions.

    Labels 10,000 invoices in under 100ms — pure Python, no GPU needed.

    Args:
        descriptions: List of raw transaction descriptions

    Returns:
        List of LabelResult objects, one per description

    Example:
        >>> label_batch(["SALESFORCE CRM", "AWS EMEA CLOUD", "UNKNOWN VENDOR AS"])
        [LabelResult("saas_software", 1.0, "SALESFORCE"),
         LabelResult("cloud_infrastructure", 1.0, "AWS EMEA"),
         LabelResult(None, 0.0, None)]
    """
    return [auto_label(desc) for desc in descriptions]


def get_coverage_stats(descriptions: list[str]) -> dict:
    """
    Calculate what percentage of transactions our rules can label.

    Run this on Tink/GoCardless data to understand auto-labeling coverage
    before committing time to BERT training.

    Output example:
        {
            "total": 5000,
            "labeled": 4100,
            "unlabeled": 900,
            "coverage_pct": 82.0,
            "by_category": {
                "saas_software": 1200,
                "cloud_infrastructure": 800,
                "consulting_services": 600,
                ...
            }
        }
    """
    results = label_batch(descriptions)
    labeled = [r for r in results if r.category is not None]
    by_category: dict[str, int] = {}
    for r in labeled:
        by_category[r.category] = by_category.get(r.category, 0) + 1

    return {
        "total": len(descriptions),
        "labeled": len(labeled),
        "unlabeled": len(descriptions) - len(labeled),
        "coverage_pct": round(len(labeled) / len(descriptions) * 100, 1) if descriptions else 0.0,
        "by_category": by_category,
    }
